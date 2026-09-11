// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use tokio::sync::Notify;
use tracing::{debug, info, warn};

use spur_core::account_limits::{
    check_account_limits_with_grp_node_charge, check_account_standalone_limits,
    check_account_submit_limits, check_account_wall_limit, AccountCheckResult,
};
use spur_core::accounting::{
    AccountLimits, PerUserCaps, Qos, ScopeLimitUsage, TresRecord, TresType, UserLimitUsage,
};
use spur_core::burst_buffer::BbStageState;
use spur_core::config::{EnforcePartLimits, HealthCheck, SlurmConfig};
use spur_core::job::{
    effective_gpus, effective_memory_mb, Job, JobId, JobSpec, JobState, NodeCompleteError,
    PendingReason, TransitionOutcome, DEFAULT_PRIORITY,
};
use spur_core::node::{Node, NodeEvent, NodeSource, NodeState};
use spur_core::partition::{requested_partition_names, Partition, PreemptMode};
use spur_core::qos::{
    check_qos_limits_with_grp_node_charge, check_qos_standalone_limits, check_qos_submit_limits,
    QosCheckResult,
};
use spur_core::reservation::{self, normalize_node_list, running_jobs_overlap_start, Reservation};
use spur_core::resource::{ResourceAllocations, ResourceSet};
use spur_core::step::{JobStep, StepState, STEP_BATCH, STEP_RESERVED_MIN};
use spur_core::wal::WalOperation;
use spur_metrics::job::JobMetricsSnapshot;
use spur_metrics::node::NodeMetricsSnapshot;
use spur_metrics::partition::PartitionMetricsSnapshot;
use spur_metrics::user_acct::UserAcctMetricsSnapshot;

use crate::accounting::{
    txn, AccountingNotifier, JobStartRecord, TxnAction, TxnEntity, TxnOutcome, TxnRecord, TxnSource,
};
use crate::association_cache::{qos_permitted, AccountMembership, AssociationCache};
use crate::fairshare_cache::FairshareCache;
use crate::limits_cache::{consumed_minutes, GrpWallCache, QosCache};
use crate::pmix_dispatch;
use crate::raft::{ClientResponse, JobFinalized, SpurRaft, StateMachineApply};
use crate::sched_stats::SchedStatsCollector;

/// Description shown for a job parked after a launch failure it would carry to
/// the next node. Byte-exact with the `state_desc` Slurm sets in the same case,
/// so operator runbooks and log greps written against Slurm keep working.
pub const LAUNCH_FAILURE_HELD_DESC: &str = "launch failed requeued held";

/// Seconds to defer a job by after its `requeue_count`-th launch failure,
/// doubling per attempt from the same base the preemption requeue uses.
///
/// The shift is clamped and the result double-capped because `max_batch_requeue`
/// and the cap are both operator-supplied: an unvalidated config must degrade
/// rather than push the computed instant out of chrono's range and panic the
/// controller.
fn launch_backoff_secs(interval_secs: u32, cap: u64, requeue_count: u32) -> u64 {
    let base = (interval_secs as u64 * 2 + 3).max(5);
    base.saturating_mul(1u64 << requeue_count.min(16))
        .min(cap)
        .min(spur_core::config::MAX_LAUNCH_BACKOFF_SECS)
}

/// Result of recording a per-node completion report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeCompleteResult {
    /// Node recorded; waiting for remaining nodes.
    Completing,
    /// All allocated nodes have reported; job is now terminal.
    AllDone { state: JobState, exit_code: i32 },
    /// Job was already in a terminal state (duplicate or race with cancel/timeout).
    AlreadyTerminal,
    /// Report came from a superseded run (older `run_attempt`); ignored so it
    /// cannot fail a job that has since been requeued and re-dispatched.
    StaleReport,
}

/// Reservation CRUD errors for the gRPC boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReservationError {
    InvalidArgument(String),
    NotFound(String),
    AlreadyExists(String),
    Raft(String),
}

impl std::fmt::Display for ReservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(m)
            | Self::NotFound(m)
            | Self::AlreadyExists(m)
            | Self::Raft(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for ReservationError {}

/// Job submission errors for the gRPC / REST boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitError {
    InvalidArgument(String),
    /// A transient condition the client should retry (maps to gRPC `Unavailable`,
    /// REST 503), e.g. a cold association cache right after controller start.
    Unavailable(String),
    Internal(String),
}

/// Errors from a node (re-)registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegisterNodeError {
    PermissionDenied(String),
    Internal(String),
}

impl std::fmt::Display for RegisterNodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PermissionDenied(m) | Self::Internal(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for RegisterNodeError {}

/// Errors from completing a standalone srun allocation.
#[derive(Debug)]
pub enum SrunCompleteError {
    NotFound(JobId),
    NotSrunJob(JobId),
    NotStepDispatch(JobId),
    AlreadyTerminal { job_id: JobId, state: JobState },
    NotOwner { job_id: JobId, user: String },
    Internal { job_id: JobId, message: String },
}

impl std::fmt::Display for SrunCompleteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "job {id} not found"),
            Self::NotSrunJob(id) => write!(f, "job {id} is not an srun allocation"),
            Self::NotStepDispatch(id) => {
                write!(
                    f,
                    "job {id} does not use native step dispatch (CompleteJob is not valid)"
                )
            }
            Self::AlreadyTerminal { job_id, state } => {
                write!(f, "job {job_id} is already {state:?}")
            }
            Self::NotOwner { job_id, user } => {
                write!(f, "user {user} is not permitted to complete job {job_id}")
            }
            Self::Internal { job_id, message } => {
                write!(f, "job {job_id}: {message}")
            }
        }
    }
}

impl std::error::Error for SrunCompleteError {}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(m) | Self::Unavailable(m) | Self::Internal(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for SubmitError {}

impl SubmitError {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }

    pub fn unavailable(msg: impl Into<String>) -> Self {
        Self::Unavailable(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}

/// A successful submission: the (parent) job id plus any user-facing warnings
/// (e.g. a node-count reduction) to surface at the CLI and REST response.
#[derive(Debug, Clone, Default)]
pub struct SubmitOutcome {
    pub job_id: JobId,
    pub warnings: Vec<String>,
}

/// Maximum serialized size of a single job submission, in bytes.
///
/// A submission becomes one Raft log entry (`WalOperation::JobSubmit`) that is
/// also retained in every snapshot, so bounding the whole serialized spec keeps
/// that entry well under `crate::raft::RAFT_MAX_MESSAGE_SIZE`. Measuring the
/// serialized form rather than individual fields counts every payload the spec
/// carries (script, environment, argv, container env/mounts, ...), so it cannot
/// be bypassed by a field the check forgot to sum. Mirrors Slurm, which rejects
/// oversized batch scripts.
const MAX_JOB_SPEC_SIZE: usize = 4 * 1024 * 1024;

/// A `std::io::Write` that only tallies byte counts and discards the data. Used
/// to measure a value's serialized size without allocating the serialized bytes.
#[derive(Default)]
struct ByteCounter {
    len: usize,
}

impl std::io::Write for ByteCounter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.len += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn check_submission_size(spec: &JobSpec) -> Result<(), SubmitError> {
    // Count serialized bytes without allocating the output; the entry is
    // re-serialized by openraft on propose.
    let mut counter = ByteCounter::default();
    serde_json::to_writer(&mut counter, spec)
        .map_err(|e| SubmitError::internal(format!("failed to encode job spec: {e}")))?;
    let size = counter.len;
    if size > MAX_JOB_SPEC_SIZE {
        return Err(SubmitError::invalid(format!(
            "job submission too large: {:.1} MiB serialized, limit is {} MiB (reduce script, environment, or argv size)",
            size as f64 / (1024.0 * 1024.0),
            MAX_JOB_SPEC_SIZE / (1024 * 1024),
        )));
    }
    Ok(())
}

impl ReservationError {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    pub fn already_exists(msg: impl Into<String>) -> Self {
        Self::AlreadyExists(msg.into())
    }

    pub fn raft(msg: impl Into<String>) -> Self {
        Self::Raft(msg.into())
    }
}

/// Partition CRUD errors for the gRPC boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionError {
    InvalidArgument(String),
    NotFound(String),
    AlreadyExists(String),
    Raft(String),
}

impl std::fmt::Display for PartitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(m)
            | Self::NotFound(m)
            | Self::AlreadyExists(m)
            | Self::Raft(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for PartitionError {}

impl PartitionError {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    pub fn already_exists(msg: impl Into<String>) -> Self {
        Self::AlreadyExists(msg.into())
    }

    pub fn raft(msg: impl Into<String>) -> Self {
        Self::Raft(msg.into())
    }
}

/// What the caller must dispatch to node agents after `preempt_job`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptOutcome {
    /// Kill the job's processes on every allocated node (cancel or requeue mode).
    Killed,
    /// Stop (SIGSTOP) the job's processes; allocation is retained (suspend mode).
    Suspended,
}

/// Per-role node counts backing `/metrics/k8s`'s `spur_k8s_nodes_by_role`. Kept in sync with
/// `Node.k0s_role` incrementally in `apply()` (every replica, so a promoted follower is already
/// correct); `recompute_from` is the only O(n) path, used once on snapshot restore/startup.
#[derive(Default)]
struct K0sRoleCounts {
    controller: AtomicU64,
    worker: AtomicU64,
    single: AtomicU64,
}

impl K0sRoleCounts {
    fn counter(&self, role: spur_core::k0s::K0sRole) -> &AtomicU64 {
        use spur_core::k0s::K0sRole;
        match role {
            K0sRole::Controller => &self.controller,
            K0sRole::Worker => &self.worker,
            K0sRole::Single => &self.single,
        }
    }

    fn inc(&self, role: spur_core::k0s::K0sRole) {
        self.counter(role).fetch_add(1, Ordering::Relaxed);
    }

    fn dec(&self, role: spur_core::k0s::K0sRole) {
        self.counter(role).fetch_sub(1, Ordering::Relaxed);
    }

    /// (controller, worker, single).
    fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.controller.load(Ordering::Relaxed),
            self.worker.load(Ordering::Relaxed),
            self.single.load(Ordering::Relaxed),
        )
    }

    fn recompute_from(&self, roles: impl Iterator<Item = Option<spur_core::k0s::K0sRole>>) {
        use spur_core::k0s::K0sRole;
        let (mut controller, mut worker, mut single) = (0u64, 0u64, 0u64);
        for role in roles.flatten() {
            match role {
                K0sRole::Controller => controller += 1,
                K0sRole::Worker => worker += 1,
                K0sRole::Single => single += 1,
            }
        }
        self.controller.store(controller, Ordering::Relaxed);
        self.worker.store(worker, Ordering::Relaxed);
        self.single.store(single, Ordering::Relaxed);
    }
}

/// Central cluster state manager.
///
/// Thread-safe via RwLock. The scheduler and gRPC server both access this.
/// State recovery happens through Raft log replay (via `StateMachineApply`).
pub struct ClusterManager {
    /// Live cluster configuration, swapped wholesale by `reconfigure()`.
    ///
    /// Held behind `RwLock<Arc<_>>` so `reconfigure` can atomically replace it
    /// while readers take a cheap snapshot via `config()`. Sections consumed
    /// per-request or per-scheduler-cycle through that snapshot pick up new
    /// values live; sections captured once at startup (bound sockets, DB pool,
    /// scheduler loop interval) remain restart-only — see `reconfigure`.
    config: RwLock<Arc<SlurmConfig>>,
    /// Scheduler tick interval captured at startup. The scheduler loop's cadence
    /// is fixed once at boot (restart-only), so the preemption requeue hold —
    /// which is sized to that cadence — must read this pinned value, not the
    /// live `config()`, or the hold window would drift after `reconfigure`
    /// while the loop keeps ticking at the old rate.
    scheduler_interval_secs: u32,
    /// Path to the spur.conf file, re-read by `reconfigure()`. spur.conf is
    /// never written back — the Raft WAL is the sole source of runtime truth.
    /// None when running without a config file (e.g. in tests).
    config_path: Option<PathBuf>,
    jobs: RwLock<HashMap<JobId, Job>>,
    nodes: RwLock<HashMap<String, Node>>,
    partitions: RwLock<Vec<Partition>>,
    /// Names of partitions that were runtime-deleted. Used to suppress config-file
    /// partitions with the same name from re-appearing on restart.
    deleted_partition_names: RwLock<HashSet<String>>,
    /// Partition names seeded from config before WAL replay. A replayed
    /// `PartitionCreate` for one overwrites the seed (WAL is authoritative);
    /// once overridden the name is removed, so later duplicates stay
    /// first-writer-wins. Runtime-only, never persisted.
    config_seeded_partitions: RwLock<HashSet<String>>,
    next_job_id: AtomicU32,
    reservations: RwLock<Vec<Reservation>>,
    steps: RwLock<HashMap<(JobId, u32), JobStep>>,
    /// Configured cluster-wide license totals (immutable; from config). Current
    /// availability is derived as total minus the licenses held by active jobs
    /// (see `available_licenses`), so it cannot drift or diverge from config.
    license_pool: RwLock<HashMap<String, u64>>,
    /// Configured cluster-wide burst-buffer capacity in GB (immutable; from
    /// config). Like licenses, current availability is derived as total minus
    /// the capacity reserved by jobs that have entered staging (see
    /// `available_bb_with`), so it cannot drift from config.
    burst_buffer_total_gb: RwLock<u64>,
    tokens: RwLock<HashMap<String, spur_core::admission::AdmissionToken>>,
    /// Native k0s cluster state (phase, control-plane node, join-token metadata).
    k0s: RwLock<spur_core::k0s::K0sClusterState>,
    /// Long-lived k0s lifecycle/node metric accumulator exported at `/metrics/k8s`.
    k8s_metrics: Arc<spur_metrics::K8sMetrics>,
    /// Per-role node counts for `/metrics/k8s`, avoiding a full node scan every scrape.
    k0s_role_counts: K0sRoleCounts,
    /// Serializes k0s phase-transition accounting so concurrent `set_k0s_phase` callers can't both
    /// read the same prior phase and double-count the edge.
    k0s_phase_accounting: parking_lot::Mutex<()>,
    /// Per-node locks serializing (re-)registration, so unrelated nodes don't block each other
    /// while a same-name racer can't act on a stale label diff (see `register_node`).
    node_registration_locks: parking_lot::Mutex<HashMap<String, Arc<parking_lot::Mutex<()>>>>,
    raft: RwLock<Option<SpurRaft>>,
    accounting: RwLock<Option<AccountingNotifier>>,
    fairshare_cache: Arc<FairshareCache>,
    qos_cache: Arc<QosCache>,
    grp_wall_cache: Arc<GrpWallCache>,
    association_cache: Arc<AssociationCache>,
    /// Wake signal for the scheduler loop.
    pub(crate) scheduler_notify: Arc<Notify>,
    sched_stats: OnceLock<Arc<SchedStatsCollector>>,
    /// Node -> (job_id, start_time) for a future backfill reservation,
    /// refreshed every cycle. Ephemeral: never persisted, cheap to recompute.
    planned_reservations: RwLock<HashMap<String, (JobId, DateTime<Utc>)>>,
    /// Same scheduler plan keyed by job, including slots on busy nodes that
    /// `planned_reservations` (an idle-node view for sinfo) leaves out.
    planned_job_starts: RwLock<spur_sched::backfill::PlannedJobStarts>,
    /// Last keepalive time per interactive allocation, used by the InactiveLimit
    /// reaper. Ephemeral soft state (like `Node::last_heartbeat`): keepalives
    /// arrive too often to persist, and on failover the reaper reseeds lazily.
    interactive_last_seen: RwLock<HashMap<JobId, DateTime<Utc>>>,
    /// Nodes skipped for new dispatch until the given instant after a
    /// resources-unavailable reject. Leader-local and transient, never persisted.
    node_dispatch_cooldowns: RwLock<HashMap<String, std::time::Instant>>,
    /// When each (check index, node name) last completed a check, so the pass
    /// knows when the next one is due. Leader-local and transient (like
    /// `node_dispatch_cooldowns`): reset on failover, which at worst re-runs one
    /// round of checks on the new leader. In-flight and drain state are read
    /// from the durable job store / node reasons instead, so they survive
    /// failover.
    health_last_check: parking_lot::Mutex<HashMap<(usize, String), std::time::Instant>>,
}

/// Reserved job-name prefix marking a controller-submitted health-check job, so
/// its finalization can be recognized and users can't spoof one. The full name
/// is `{PREFIX}{check_index}.{node}`.
pub(crate) const HEALTH_JOB_PREFIX: &str = "_spur-health.";

/// Node `state_reason` prefix for a node the health system soft-drained *to make
/// room* for a starved check. Such a node auto-resumes once it empties so the
/// pending check can land — a property read from this durable reason, not the
/// leader-local slot, so it holds across a failover.
const HEALTH_FORCE_DRAIN_PREFIX: &str = "health-check(pending): ";

/// Node `state_reason` prefix for a node a check actually *failed* on. This
/// drain is sticky: the node stays out of service for an operator to inspect,
/// exactly like Slurm's failed `HealthCheckProgram`. It never auto-resumes.
const HEALTH_FAIL_DRAIN_PREFIX: &str = "health-check(failed): ";

/// The node a health-check job targets, parsed from its reserved name
/// (`{HEALTH_JOB_PREFIX}{idx}.{node}`). `None` for any non-health name.
fn health_job_target_node(name: &str) -> Option<&str> {
    name.strip_prefix(HEALTH_JOB_PREFIX)
        .and_then(|rest| rest.split_once('.'))
        .map(|(_, node)| node)
}

struct PendingJobClassification {
    jobs: Vec<Job>,
    reason_updates: Vec<(JobId, PendingReason)>,
    bb_stage_candidates: Vec<JobId>,
}

struct PendingJobCandidate {
    job: Job,
    scheduling_eligible: bool,
    tag_reason: bool,
}

/// Filters for [`ClusterManager::get_jobs`]. Every field is optional: an empty
/// slice or `None` disables that dimension, so `JobFilter::default()` returns
/// all jobs.
#[derive(Debug, Default, Clone)]
pub struct JobFilter<'a> {
    pub states: &'a [JobState],
    pub user: Option<&'a str>,
    pub partition: Option<&'a str>,
    pub account: Option<&'a str>,
    pub name: Option<&'a str>,
    pub job_ids: &'a [JobId],
    /// Concrete node names (already hostlist-expanded); keeps only jobs
    /// allocated on at least one of them.
    pub nodes: &'a [String],
}

/// Result of an admin requeue: `killed` are Running/Suspended snapshots whose
/// old processes must be cancelled; `skipped` lists array tasks not requeued.
#[derive(Debug, Default)]
pub struct RequeueOutcome {
    pub requeued: u32,
    pub killed: Vec<Job>,
    pub skipped: Vec<String>,
}

impl ClusterManager {
    #[cfg(test)]
    pub fn new(config: SlurmConfig, state_dir: &Path) -> anyhow::Result<Self> {
        Self::new_with_config_path(config, state_dir, None)
    }

    pub fn new_with_config_path(
        config: SlurmConfig,
        _state_dir: &Path,
        config_path: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let partitions = config.build_partitions();
        let config_seeded_partitions: HashSet<String> =
            partitions.iter().map(|p| p.name.clone()).collect();
        let license_pool = config.licenses.clone();
        let burst_buffer_total_gb = config.burst_buffer.total_gb;
        let fairshare_cache = Arc::new(FairshareCache::new());
        let first_job_id = config.controller.first_job_id;
        let scheduler_interval_secs = config.scheduler.interval_secs;
        let qos_cache = Arc::new(QosCache::new());
        let grp_wall_cache = Arc::new(GrpWallCache::new());
        let association_cache = Arc::new(AssociationCache::new());

        let cm = Self {
            config: RwLock::new(Arc::new(config)),
            scheduler_interval_secs,
            config_path,
            jobs: RwLock::new(HashMap::new()),
            nodes: RwLock::new(HashMap::new()),
            partitions: RwLock::new(partitions),
            deleted_partition_names: RwLock::new(HashSet::new()),
            config_seeded_partitions: RwLock::new(config_seeded_partitions),
            reservations: RwLock::new(Vec::new()),
            steps: RwLock::new(HashMap::new()),
            next_job_id: AtomicU32::new(first_job_id),
            license_pool: RwLock::new(license_pool),
            burst_buffer_total_gb: RwLock::new(burst_buffer_total_gb),
            tokens: RwLock::new(HashMap::new()),
            k0s: RwLock::new(spur_core::k0s::K0sClusterState::default()),
            k8s_metrics: Arc::new(spur_metrics::K8sMetrics::new()),
            k0s_role_counts: K0sRoleCounts::default(),
            k0s_phase_accounting: parking_lot::Mutex::new(()),
            node_registration_locks: parking_lot::Mutex::new(HashMap::new()),
            raft: RwLock::new(None),
            accounting: RwLock::new(None),
            fairshare_cache,
            qos_cache,
            grp_wall_cache,
            association_cache,
            scheduler_notify: Arc::new(Notify::new()),
            sched_stats: OnceLock::new(),
            planned_reservations: RwLock::new(HashMap::new()),
            planned_job_starts: RwLock::new(HashMap::new()),
            interactive_last_seen: RwLock::new(HashMap::new()),
            node_dispatch_cooldowns: RwLock::new(HashMap::new()),
            health_last_check: parking_lot::Mutex::new(HashMap::new()),
        };

        info!("cluster manager initialized (state will be recovered via Raft)");

        Ok(cm)
    }

    /// Snapshot the live configuration. Cheap `Arc` clone; callers read fields
    /// off the returned snapshot so a concurrent `reconfigure` swap is atomic
    /// from their point of view.
    pub fn config(&self) -> Arc<SlurmConfig> {
        self.config.read().clone()
    }

    /// Skip a node for new dispatch for the configured cooldown after it rejected a
    /// launch or could not be reached, so the scheduler stops re-picking it each tick.
    pub fn cool_down_node(&self, name: &str) {
        let secs = self.config().controller.dispatch_reject_cooldown_secs;
        if secs == 0 {
            return;
        }
        self.cool_down_node_for(name, std::time::Duration::from_secs(secs));
    }

    /// Cool down for an explicit span. A node that burned a whole dispatch deadline must not be
    /// re-picked after the much shorter reject cooldown, or it spends most of every cycle stalling.
    pub fn cool_down_node_for(&self, name: &str, span: std::time::Duration) {
        if span.is_zero() {
            return;
        }
        let until = std::time::Instant::now() + span;
        self.node_dispatch_cooldowns
            .write()
            .insert(name.to_string(), until);
    }

    /// How long a node's dispatch cooldown still has to run, for tests that need to tell the
    /// reject cooldown apart from the longer dispatch-deadline one.
    #[cfg(test)]
    pub fn dispatch_cooldown_remaining(&self, name: &str) -> Option<std::time::Duration> {
        let now = std::time::Instant::now();
        self.node_dispatch_cooldowns
            .read()
            .get(name)
            .map(|&until| until.saturating_duration_since(now))
    }

    /// Names still within their dispatch cooldown, pruning any that have expired.
    pub fn nodes_on_dispatch_cooldown(&self) -> HashSet<String> {
        let now = std::time::Instant::now();
        let mut cooldowns = self.node_dispatch_cooldowns.write();
        cooldowns.retain(|_, &mut until| until > now);
        cooldowns.keys().cloned().collect()
    }

    /// Submit a new job. If it has an array spec, expand into individual tasks.
    pub fn submit_job(&self, mut spec: JobSpec) -> Result<SubmitOutcome, SubmitError> {
        // The health-check subsystem owns this name prefix; reject it on the
        // user path so a submitted job can never masquerade as a health check
        // and drive a node drain/resume on completion.
        if spec.name.starts_with(HEALTH_JOB_PREFIX) {
            return Err(SubmitError::invalid(format!(
                "job name may not start with the reserved prefix '{HEALTH_JOB_PREFIX}'"
            )));
        }

        // One config/partitions snapshot for the whole submit path, so defaulting
        // and enforcement can't observe a concurrent reconfigure() mid-submit.
        let config = self.config();
        let partitions = self.partitions.read().clone();

        apply_default_partition(&mut spec, &partitions);

        // EnforcePartLimits mirrors Slurm: only a wall-time the user actually
        // requested is subject to submit-time rejection, never one we auto-fill.
        let user_requested_time = spec.time_limit.is_some();
        if let Some(tl) = spec.time_limit.as_ref() {
            if tl.num_seconds() < 0 {
                return Err(SubmitError::invalid(
                    "requested time limit must not be negative",
                ));
            }
        }

        apply_default_account(&mut spec, &self.association_cache);
        validate_user_account(&spec, &self.association_cache, &config.accounting)?;
        // Default QoS must resolve before the partition ACL, or `allow_qos` sees
        // an empty QoS and wrongly rejects a user's inherited default.
        apply_default_qos(
            &mut spec,
            &self.association_cache,
            &self.qos_cache,
            &config.accounting,
        )?;
        self.validate_partition(&spec, &partitions)?;

        // Fewer tasks than nodes cannot use the surplus nodes; cap at the task
        // count so reported node/GPU counts match the allocation (unless a
        // per-node layout is pinned).
        let requested_nodes = spec.num_nodes.max(1);
        spec.num_nodes = spec.effective_num_nodes();

        let mut warnings = Vec::new();
        if spec.num_nodes < requested_nodes {
            warn!(
                requested_nodes,
                allocated_nodes = spec.num_nodes,
                num_tasks = spec.num_tasks,
                "reduced requested node count to task count at submit"
            );
            warnings.push(format!(
                "requested {requested_nodes} nodes but only {} will be allocated \
                 ({} task(s), one per node)",
                spec.num_nodes, spec.num_tasks
            ));
        }

        // Validate GPU demand against the normalized node count so a request
        // like `-N4 -n1 --gpus=2` (valid once reduced to one node) is accepted.
        spur_core::gpu_request::resolve_gpu_demand(&spec)
            .map_err(|e| SubmitError::invalid(e.to_string()))?;

        // Reject a node count outside the partition's bounds at submit, matching
        // Slurm, instead of accepting a job that would pend forever.
        self.validate_partition_node_bounds(&spec, &partitions)?;

        // Reject an over-MaxTime wall-time at submit when EnforcePartLimits is on
        // (default off keeps Slurm's admit-and-pend behavior). Only a
        // user-requested `-t` is enforced; an auto-filled default never rejects.
        if user_requested_time {
            validate_partition_time_limit(
                &spec,
                config.scheduler.enforce_part_limits,
                &partitions,
            )?;
        }

        let mpi = spec.mpi.as_deref().unwrap_or(spur_core::mpi::MPI_NONE);
        pmix_dispatch::validate_multi_node_pmix_nodelist(
            mpi,
            spec.num_nodes,
            spec.nodelist.as_deref(),
            |name| self.nodes.read().get(name).map(|node| node.source.clone()),
        )
        .map_err(SubmitError::invalid)?;

        // Bound the spec before the hook forks/parses it; the hook only edits
        // bounded whitelist fields, so this still measures what gets persisted.
        check_submission_size(&spec)?;

        self.run_job_submit_hook(&mut spec, &partitions)?;

        // Every input to wall-time defaulting — partition, account, QOS — is a field
        // the hook may rewrite, so defaulting follows the hook and reads the scope
        // that actually governs the job. A limit the hook set is left alone.
        let wall_caps = self.wall_caps(&spec);
        apply_default_time_limit(
            &mut spec,
            &partitions,
            config.scheduler.default_time_limit_minutes,
            wall_caps,
        );

        // Reject unknown/malformed dependency types up front so users get a
        // clear error instead of a silently-deadlocked job (e.g. `expand:N`).
        // This validates syntax only — the dependency *target* is intentionally
        // not checked for existence here (matching Slurm), so e.g. `after:9999`
        // against a nonexistent job is accepted and resolves as satisfiable.
        if !spec.dependency.is_empty() {
            spur_core::dependency::try_parse_dependencies(&spec.dependency)
                .map_err(|e| SubmitError::invalid(format!("invalid dependency: {e}")))?;
        }

        // Submission gate: deny the submit-count family and (for DenyOnLimit
        // QOS) unsatisfiable stand-alone resource requests before the job ever
        // enters the queue. The scheduler path stays a safety net for the
        // limit-lowered-after-submit edge case. Each array task counts
        // individually toward submit caps; the whole submission is rejected.
        let incoming_tasks = match spec.array_spec.as_deref() {
            Some(s) => {
                let parsed = spur_core::array::parse_array_spec(s)
                    .map_err(|e| SubmitError::invalid(e.to_string()))?;
                u32::try_from(parsed.task_ids.len()).unwrap_or(u32::MAX)
            }
            None => 1,
        };
        self.enforce_submit_limits(&spec, incoming_tasks)?;

        let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);
        let specs =
            expand_job_specs(spec, job_id).map_err(|e| SubmitError::invalid(e.to_string()))?;

        for mut task_spec in specs {
            let task_id = if task_spec.array_job_id.is_some() {
                self.next_job_id.fetch_add(1, Ordering::SeqCst)
            } else {
                job_id
            };

            // Seed the base from QOS so it feeds the multiplicative formula rather than
            // being a negligible delta on top of it.
            if task_spec.priority.is_none() {
                if let Some(qos_name) = task_spec.qos.as_deref() {
                    if let Some(qos) = self.qos_cache.get(qos_name) {
                        task_spec.priority =
                            Some(DEFAULT_PRIORITY.saturating_add_signed(qos.priority).max(1));
                    }
                }
            }

            self.propose(WalOperation::JobSubmit {
                job_id: task_id,
                spec: Box::new(task_spec),
            })
            .map_err(|e| SubmitError::internal(e.to_string()))?;
            if let Some(stats) = self.sched_stats.get() {
                stats.record_submitted(1);
            }
        }

        self.scheduler_notify.notify_one();

        info!(job_id, "job submitted");
        Ok(SubmitOutcome { job_id, warnings })
    }

    /// Reject a submission whose (normalized) node count falls outside the
    /// bounds of every requested partition. Matches Slurm, which rejects such
    /// jobs at submit rather than leaving them permanently pending. A partition
    /// list is accepted if any one partition can hold the request.
    fn validate_partition_node_bounds(
        &self,
        spec: &JobSpec,
        partitions: &[Partition],
    ) -> Result<(), SubmitError> {
        let Some(partition_spec) = spec.partition.as_deref().filter(|p| !p.is_empty()) else {
            return Ok(());
        };

        let requested: Vec<&Partition> = requested_partition_names(Some(partition_spec))
            .filter_map(|name| partitions.iter().find(|part| part.name == name))
            .collect();
        // Existence was already checked in validate_partition.
        if requested.is_empty() {
            return Ok(());
        }

        let nodes = spec.num_nodes;
        let fits = |part: &Partition| {
            let below_min = part.min_nodes > 0 && nodes < part.min_nodes;
            let above_max = part.max_nodes.is_some_and(|max| nodes > max);
            !below_min && !above_max
        };
        if requested.iter().copied().any(fits) {
            return Ok(());
        }

        let part = requested[0];
        let max = part
            .max_nodes
            .map(|m| m.to_string())
            .unwrap_or_else(|| "unlimited".into());
        Err(SubmitError::invalid(format!(
            "requested node count {nodes} is outside partition '{}' limits (min {}, max {})",
            part.name, part.min_nodes, max
        )))
    }

    /// Run the configured job-submission hooks against the resolved spec: reject
    /// maps to an invalid-argument error; modify applies whitelisted changes.
    /// The shell hook runs first, then the Lua hook, each on the evolving spec
    /// (mirroring Slurm's config-ordered plugin chain).
    fn run_job_submit_hook(
        &self,
        spec: &mut JobSpec,
        partitions: &[Partition],
    ) -> Result<(), SubmitError> {
        let config = self.config();
        let shell = config.hooks.job_submit.as_deref();
        let lua = config.hooks.job_submit_lua.as_deref();
        if shell.is_none() && lua.is_none() {
            return Ok(());
        }

        if let Some(script) = shell {
            let ctx = self.submit_hook_ctx(spec)?;
            let outcome = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(async { spur_core::hooks::run_submit_hook(script, &ctx).await })
            })
            .map_err(|e| SubmitError::internal(format!("job_submit hook failed: {e}")))?;
            self.apply_submit_outcome(spec, "job_submit", outcome, partitions)?;
        }

        if let Some(script) = lua {
            let ctx = self.submit_hook_ctx(spec)?;
            // The Lua VM runs synchronously; block_in_place keeps it off the async
            // worker, matching the shell path and the `propose()` convention.
            let outcome =
                tokio::task::block_in_place(|| crate::hooks::run_submit_hook_lua(script, &ctx))
                    .map_err(|e| {
                        SubmitError::internal(format!("job_submit lua hook failed: {e}"))
                    })?;
            self.apply_submit_outcome(spec, "job_submit_lua", outcome, partitions)?;
        }

        Ok(())
    }

    fn submit_hook_ctx(
        &self,
        spec: &JobSpec,
    ) -> Result<spur_core::hooks::SubmitHookContext, SubmitError> {
        let spec_json = serde_json::to_string(spec).map_err(|e| {
            SubmitError::internal(format!(
                "failed to encode job spec for job_submit hook: {e}"
            ))
        })?;
        Ok(spur_core::hooks::SubmitHookContext {
            spec_json,
            user: spec.user.clone(),
            uid: spec.uid,
            gid: spec.gid,
            partition: spec.partition.clone().unwrap_or_default(),
        })
    }

    /// Apply one hook's outcome: reject fails the submission; modify applies
    /// whitelisted changes, then re-validates partition/account ACLs and QOS.
    fn apply_submit_outcome(
        &self,
        spec: &mut JobSpec,
        hook: &str,
        outcome: spur_core::hooks::SubmitHookOutcome,
        partitions: &[Partition],
    ) -> Result<(), SubmitError> {
        match outcome {
            spur_core::hooks::SubmitHookOutcome::Accept => {
                info!(target: "audit", hook, user = %spec.user, uid = spec.uid, "job_submit hook accepted");
                Ok(())
            }
            spur_core::hooks::SubmitHookOutcome::Reject(reason) => {
                info!(target: "audit", hook, user = %spec.user, uid = spec.uid, reason = %reason, "job_submit hook rejected");
                Err(SubmitError::invalid(reason))
            }
            spur_core::hooks::SubmitHookOutcome::Modify(changes) => {
                let modified = spur_core::hooks::apply_submit_changes(spec, &changes);
                // An empty change set is an accept, not a modify (no bogus audit line).
                if modified.is_empty() {
                    info!(target: "audit", hook, user = %spec.user, uid = spec.uid, "job_submit hook accepted");
                    return Ok(());
                }
                // comment/constraint/gres are unbounded strings; re-enforce the
                // Raft entry size cap the hook could otherwise grow past.
                check_submission_size(spec)?;
                info!(
                    target: "audit",
                    hook,
                    user = %spec.user,
                    uid = spec.uid,
                    partition = %spec.partition.clone().unwrap_or_default(),
                    gpus = spur_core::job::effective_gpus(spec, spec.num_nodes),
                    modified = ?modified,
                    "job_submit hook modified spec"
                );
                if changes.partition.is_some() || changes.account.is_some() {
                    validate_user_account(
                        spec,
                        &self.association_cache,
                        &self.config().accounting,
                    )?;
                    self.validate_partition_node_bounds(spec, partitions)?;
                }
                // Existence is enforced first: an unknown QOS silently resolves
                // to the limitless default, which would bypass QOS limits.
                if let Some(name) = changes.qos.as_deref().filter(|n| !n.is_empty()) {
                    if self.qos_cache.get(name).is_none() {
                        return Err(SubmitError::invalid(format!(
                            "job_submit hook set unknown QOS '{name}'"
                        )));
                    }
                }
                // partition/account/qos all feed the partition's allow_qos/deny_qos
                // ACL and the submitter's association-level authorization, so any of
                // the three re-checks both against the already-applied spec.
                if changes.partition.is_some() || changes.account.is_some() || changes.qos.is_some()
                {
                    self.validate_partition(spec, partitions)?;
                    if let Some(qos) = spec.qos.as_deref().filter(|q| !q.is_empty()) {
                        if let Some(account) = spec.account.as_deref().filter(|a| !a.is_empty()) {
                            self.association_cache
                                .check_qos_authorized(&spec.user, account, qos)
                                .map_err(SubmitError::invalid)?;
                        }
                    }
                }
                // A hook-set gres conflicting with an explicit --gpus request would
                // otherwise surface only at schedule time; reject it at submit.
                if changes.gres.is_some() {
                    spur_core::gpu_request::resolve_gpu_demand_for(spec, spec.num_nodes).map_err(
                        |e| {
                            SubmitError::invalid(format!(
                                "job_submit hook set a conflicting gres request: {e}"
                            ))
                        },
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Validate partition constraints: access control and node limits.
    pub(crate) fn validate_partition(
        &self,
        spec: &JobSpec,
        partitions: &[Partition],
    ) -> Result<(), SubmitError> {
        let partition_spec = match spec.partition.as_deref() {
            Some(p) if !p.is_empty() => p,
            _ => return Ok(()), // Unset or empty partition name — nothing to validate
        };

        let requested = requested_partition_names(Some(partition_spec))
            .map(|name| {
                partitions
                    .iter()
                    .find(|part| part.name == name)
                    .ok_or_else(|| SubmitError::invalid(format!("partition '{name}' not found")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if requested.is_empty() {
            return Err(SubmitError::invalid(format!(
                "partition '{partition_spec}' not found"
            )));
        }

        for part in &requested {
            if !part.allow_qos.is_empty() {
                match spec.qos.as_deref().filter(|q| !q.is_empty()) {
                    Some(qos) if part.allow_qos.iter().any(|q| q == qos) => {}
                    Some(qos) => {
                        return Err(SubmitError::invalid(format!(
                            "QoS '{qos}' not allowed on partition '{}' (allowed: {})",
                            part.name,
                            part.allow_qos.join(", ")
                        )));
                    }
                    None => {
                        return Err(SubmitError::invalid(format!(
                            "a QoS is required on partition '{}' (allowed: {})",
                            part.name,
                            part.allow_qos.join(", ")
                        )));
                    }
                }
            }
            if let Some(ref qos) = spec.qos {
                if part.deny_qos.iter().any(|q| q == qos) {
                    return Err(SubmitError::invalid(format!(
                        "QoS '{qos}' denied on partition '{}'",
                        part.name
                    )));
                }
            }
        }

        let needs_acl = requested
            .iter()
            .any(|part| !part.allow_accounts.is_empty() || !part.deny_accounts.is_empty());
        if !needs_acl {
            return Ok(());
        }

        let account = match spec.account.as_deref().filter(|a| !a.is_empty()) {
            Some(a) => a,
            None => {
                return Err(SubmitError::invalid(format!(
                    "no account for user '{}' on partition '{partition_spec}'",
                    spec.user
                )));
            }
        };

        for part in requested {
            if !part.allow_accounts.is_empty() {
                if part.allow_accounts.iter().any(|a| a == account) {
                    continue;
                }
                return Err(SubmitError::invalid(format!(
                    "account '{account}' not allowed on partition '{}'",
                    part.name
                )));
            }
            if part.deny_accounts.iter().any(|a| a == account) {
                return Err(SubmitError::invalid(format!(
                    "account '{account}' denied on partition '{}'",
                    part.name
                )));
            }
        }

        Ok(())
    }

    /// Get a job by ID.
    pub fn get_job(&self, job_id: JobId) -> Option<Job> {
        self.jobs.read().get(&job_id).cloned()
    }

    /// Next id this controller would assign. Ids at or above it were never
    /// issued here; ids below it were not necessarily issued either.
    pub fn peek_next_job_id(&self) -> JobId {
        self.next_job_id.load(Ordering::Relaxed)
    }

    /// Get a job by ID, synthesizing an aggregate record for an array *parent*
    /// id (which has no stored job — Spur stores only per-task jobs) so
    /// `scontrol show job <array_parent>` matches Slurm instead of returning
    /// empty. The synthesized job borrows the first task's spec, reports the
    /// aggregate state, earliest start / latest end; it is never stored.
    pub fn get_job_for_display(&self, job_id: JobId) -> Option<Job> {
        let jobs = self.jobs.read();
        if let Some(j) = jobs.get(&job_id) {
            return Some(j.clone());
        }
        // Maybe it's an array parent id.
        let mut tasks: Vec<&Job> = jobs
            .values()
            .filter(|j| j.spec.array_job_id == Some(job_id))
            .collect();
        if tasks.is_empty() {
            return None;
        }
        tasks.sort_by_key(|t| t.spec.array_task_id);

        let first = tasks[0];
        let mut synth = (*first).clone();
        synth.job_id = job_id;
        // Present as the parent: drop per-task id, keep array linkage.
        synth.spec.array_task_id = None;
        synth.spec.array_job_id = Some(job_id);

        let states: Vec<JobState> = tasks.iter().map(|t| t.state).collect();
        synth.state = spur_core::array::aggregate_array_state(&states).unwrap_or(JobState::Pending);
        synth.start_time = tasks.iter().filter_map(|t| t.start_time).min();
        synth.end_time = if synth.state.is_terminal() {
            tasks.iter().filter_map(|t| t.end_time).max()
        } else {
            None
        };
        // Worst non-zero exit across tasks; None while non-terminal so a
        // pending aggregate doesn't read as "0 / success".
        synth.exit_code = if synth.state.is_terminal() {
            tasks
                .iter()
                .filter_map(|t| t.exit_code)
                .filter(|c| *c != 0)
                .max()
                .or(Some(0))
        } else {
            None
        };
        Some(synth)
    }

    /// Aggregated job metrics from the current in-memory job map (lazy scan).
    ///
    /// The `jobs` map is authoritative (WAL-backed); this scans it on each call.
    pub fn job_metrics(&self) -> JobMetricsSnapshot {
        let jobs = self.jobs.read();
        JobMetricsSnapshot::collect(jobs.values())
    }

    /// Aggregated node metrics from the current in-memory node map (lazy scan).
    ///
    /// The `nodes` map is authoritative (WAL-backed for node catalog fields);
    /// this scans it on each call.
    pub fn node_metrics(&self) -> NodeMetricsSnapshot {
        let nodes = self.nodes.read();
        NodeMetricsSnapshot::collect(nodes.values())
    }

    /// Aggregated per-partition metrics from the current job, node, and partition maps.
    pub fn partition_metrics(&self) -> PartitionMetricsSnapshot {
        let names: Vec<String> = self
            .partitions
            .read()
            .iter()
            .map(|p| p.name.clone())
            .collect();
        let jobs = self.jobs.read();
        let nodes = self.nodes.read();
        PartitionMetricsSnapshot::collect(
            names.iter().map(|s| s.as_str()),
            jobs.values(),
            nodes.values(),
        )
    }

    /// Aggregated per-user and per-account job metrics from the current job map.
    pub fn user_acct_metrics(&self) -> UserAcctMetricsSnapshot {
        let jobs = self.jobs.read();
        UserAcctMetricsSnapshot::collect(jobs.values())
    }

    /// Get jobs matching filters.
    pub fn get_jobs(&self, filter: &JobFilter) -> Vec<Job> {
        // Built once per call so the per-job node check is O(allocated_nodes)
        // rather than O(allocated_nodes * filter.nodes) for large hostlists.
        let node_set: std::collections::HashSet<&str> =
            filter.nodes.iter().map(String::as_str).collect();

        let matches = |j: &Job| -> bool {
            if !filter.states.is_empty() && !filter.states.contains(&j.state) {
                return false;
            }
            if !node_set.is_empty()
                && !j
                    .allocated_nodes
                    .iter()
                    .any(|n| node_set.contains(n.as_str()))
            {
                return false;
            }
            if let Some(u) = filter.user {
                if !u.is_empty() && j.spec.user != u {
                    return false;
                }
            }
            if let Some(p) = filter.partition {
                if !p.is_empty() && j.spec.partition.as_deref() != Some(p) {
                    return false;
                }
            }
            if let Some(a) = filter.account {
                if !a.is_empty() && j.spec.account.as_deref() != Some(a) {
                    return false;
                }
            }
            if let Some(n) = filter.name {
                if !n.is_empty() && !n.split(',').any(|pat| pat.trim() == j.spec.name) {
                    return false;
                }
            }
            true
        };

        let mut result: Vec<Job> = {
            let jobs = self.jobs.read();
            jobs.values()
                .filter(|j| {
                    if !filter.job_ids.is_empty() && !filter.job_ids.contains(&j.job_id) {
                        return false;
                    }
                    matches(j)
                })
                .cloned()
                .collect()
        };

        // Requested ids with no stored job may be array parents — synthesize
        // their aggregate. Read lock above is released before get_job_for_display.
        if !filter.job_ids.is_empty() {
            for &id in filter.job_ids {
                if result.iter().any(|j| j.job_id == id) {
                    continue;
                }
                if let Some(parent) = self.get_job_for_display(id) {
                    if matches(&parent) {
                        result.push(parent);
                    }
                }
            }
        }

        result
    }

    /// Mark a pending job as DEADLINE (Slurm parity for `--deadline`).
    ///
    /// Only valid from `Pending`: returns `Err` if the job is unknown, already
    /// terminal, or has started running. Callers treat the error as non-fatal.
    pub fn deadline_job(&self, job_id: JobId) -> anyhow::Result<()> {
        {
            let mut jobs = self.jobs.write();
            let job = jobs
                .get_mut(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            if job.state.is_terminal() {
                anyhow::bail!("job {} is already {:?}", job_id, job.state);
            }
            if job.state != JobState::Pending {
                anyhow::bail!(
                    "job {} not eligible for DEADLINE from state {:?}",
                    job_id,
                    job.state
                );
            }
            // Record the reason before the terminal transition so any
            // observer (history, audit log, late `squeue` poll) sees DeadLine
            // instead of whatever update_pending_reasons last wrote.
            job.set_pending_reason(PendingReason::DeadLine);
        }

        let resp = self.propose(WalOperation::JobComplete {
            job_id,
            exit_code: -1,
            state: JobState::Deadline,
        })?;
        self.run_all_finalized_side_effects(&resp);

        info!(job_id, "job deadline passed — transitioned to DEADLINE");
        Ok(())
    }

    /// Check that `user` is allowed to perform `action` on a job owned by `owner`.
    ///
    /// These are control-plane paths where `user` has already been bound to the verified identity
    /// (or is a daemon-internal call that leaves it empty). An empty `user` is the daemon caller and
    /// a literal `"root"` is the admin override, so both are treated as internal here; the raw
    /// [`spur_core::auth::check_job_owner`] no longer infers that itself.
    fn check_job_owner(user: &str, owner: &str, action: &str) -> anyhow::Result<()> {
        let is_internal = user.is_empty() || user == "root";
        spur_core::auth::check_job_owner(user, is_internal, owner, action).map_err(Into::into)
    }

    /// Cancel a job. The requesting `user` must be the job owner, root, or
    /// empty (trusted internal/daemon calls).
    pub fn cancel_job(&self, job_id: JobId, user: &str) -> anyhow::Result<()> {
        {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            if job.state.is_terminal() {
                anyhow::bail!("job {} is already {:?}", job_id, job.state);
            }
            Self::check_job_owner(user, &job.spec.user, "cancel")?;
        }

        // Use JobComplete (not JobStateChange) so that resource deallocation
        // fires for any allocated nodes. For pending jobs, allocated_nodes is empty
        // so the deallocation loop is a no-op.
        let resp = self.propose(WalOperation::JobComplete {
            job_id,
            exit_code: -1,
            state: JobState::Cancelled,
        })?;
        self.run_all_finalized_side_effects(&resp);

        info!(job_id, "job cancelled");
        Ok(())
    }

    /// Admin requeue (`scontrol requeue` / `requeuehold`), keeping the `job_id`.
    /// A bare array-parent id (no job record of its own) fans out to every task.
    pub fn requeue_job_by_user(
        &self,
        job_id: JobId,
        user: &str,
        caller_is_admin: bool,
        hold: bool,
    ) -> anyhow::Result<RequeueOutcome> {
        let (targets, is_array) = {
            let jobs = self.jobs.read();
            if jobs.contains_key(&job_id) {
                (vec![job_id], false)
            } else {
                let mut tasks: Vec<JobId> = jobs
                    .values()
                    .filter(|j| j.spec.array_job_id == Some(job_id))
                    .map(|j| j.job_id)
                    .collect();
                tasks.sort_unstable();
                (tasks, true)
            }
        };
        if targets.is_empty() {
            anyhow::bail!("job {} not found", job_id);
        }

        // A single, explicitly named job propagates its error verbatim so the
        // user sees exactly why (not owner, interactive, already pending, ...).
        if !is_array {
            let id = targets[0];
            let (requeued, killed) = self.requeue_one_job(id, user, caller_is_admin, hold)?;
            return Ok(if requeued {
                RequeueOutcome {
                    requeued: 1,
                    killed: killed.into_iter().collect(),
                    skipped: Vec::new(),
                }
            } else {
                // Raced to a non-requeuable state after the pre-check; report it
                // honestly rather than claiming success.
                RequeueOutcome {
                    requeued: 0,
                    killed: Vec::new(),
                    skipped: vec![format!("job {id}: state changed before requeue")],
                }
            });
        }

        // Array fan-out: requeue every task that can be, collecting per-task
        // failures. Only fail outright if no task could be requeued at all.
        let total = targets.len();
        let mut outcome = RequeueOutcome::default();
        for id in targets {
            match self.requeue_one_job(id, user, caller_is_admin, hold) {
                Ok((true, killed)) => {
                    outcome.requeued += 1;
                    outcome.killed.extend(killed);
                }
                Ok((false, _)) => outcome
                    .skipped
                    .push(format!("job {id}: state changed before requeue")),
                Err(e) => outcome.skipped.push(format!("job {id}: {e}")),
            }
        }
        if outcome.requeued == 0 && outcome.skipped.len() == total {
            anyhow::bail!(
                "no tasks of array {} could be requeued: {}",
                job_id,
                outcome.skipped.join("; ")
            );
        }
        Ok(outcome)
    }

    /// Requeue exactly one job/task. Returns `(requeued, killed)`: `requeued` is
    /// false if the job raced to a non-requeuable state between the pre-check and
    /// the apply (nothing changed); `killed` is the pre-requeue snapshot when it
    /// was live and its processes must be cancelled on the nodes.
    fn requeue_one_job(
        &self,
        job_id: JobId,
        user: &str,
        caller_is_admin: bool,
        hold: bool,
    ) -> anyhow::Result<(bool, Option<Job>)> {
        let snapshot = {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            spur_core::auth::check_job_owner(user, caller_is_admin, &job.spec.user, "requeue")
                .map_err(anyhow::Error::from)?;
            // Interactive/srun allocations have no batch script to re-run.
            if job.spec.interactive || job.spec.srun_job || job.spec.pty {
                anyhow::bail!("job {} is interactive and cannot be requeued", job_id);
            }
            match job.state {
                JobState::Pending => anyhow::bail!("job {} is already pending", job_id),
                JobState::Completing => {
                    anyhow::bail!("job {} is completing; cannot requeue", job_id)
                }
                JobState::Preempted | JobState::Requeued => {
                    anyhow::bail!("job {} is already being requeued", job_id)
                }
                _ => {}
            }
            matches!(job.state, JobState::Running | JobState::Suspended).then(|| job.clone())
        };

        // Defer a live, non-held requeue's eligibility so the scheduler can't
        // re-dispatch it into its own in-flight cancel (same guard as preemption
        // requeue). Held/terminal requeues can't hit that race. Computed on the
        // leader so every replica applies one instant.
        let begin_time = match (&snapshot, hold) {
            (Some(job), false) => {
                let hold_secs = (self.scheduler_interval_secs as i64 * 2 + 3).max(5);
                let deferred = Utc::now() + chrono::Duration::seconds(hold_secs);
                Some(
                    job.spec
                        .begin_time
                        .map_or(deferred, |user_begin| user_begin.max(deferred)),
                )
            }
            _ => None,
        };

        let resp = self.propose(WalOperation::JobUserRequeue {
            job_id,
            hold,
            begin_time,
        })?;
        // Fires accounting-end + epilog for the finalized run (live path only;
        // an already-terminal job emits no finalized entry here).
        self.run_all_finalized_side_effects(&resp);

        // The apply arm NoOps if the job raced out of a requeuable state (e.g. to
        // Completing) in the propose window; confirm it really re-pended before
        // reporting success or cancelling. Read before notify so no re-dispatch yet.
        let repended = self
            .jobs
            .read()
            .get(&job_id)
            .is_some_and(|j| j.state == JobState::Pending);
        if !repended {
            debug!(job_id, "requeue raced a concurrent state change; no-op");
            return Ok((false, None));
        }
        self.scheduler_notify.notify_one();
        info!(job_id, hold, "job requeued by admin");
        Ok((true, snapshot))
    }

    /// Complete a standalone srun allocation after its step finishes.
    pub fn finish_srun_job(
        &self,
        job_id: JobId,
        exit_code: i32,
        user: &str,
    ) -> Result<Job, SrunCompleteError> {
        let job = {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or(SrunCompleteError::NotFound(job_id))?;
            if !job.spec.srun_job {
                return Err(SrunCompleteError::NotSrunJob(job_id));
            }
            if !job.srun_step_dispatch {
                return Err(SrunCompleteError::NotStepDispatch(job_id));
            }
            if job.state.is_terminal() {
                return Err(SrunCompleteError::AlreadyTerminal {
                    job_id,
                    state: job.state,
                });
            }
            Self::check_job_owner(user, &job.spec.user, "complete").map_err(|_| {
                SrunCompleteError::NotOwner {
                    job_id,
                    user: user.to_string(),
                }
            })?;
            job.clone()
        };

        let state = if exit_code == 0 {
            JobState::Completed
        } else {
            JobState::Failed
        };
        self.complete_job(job_id, exit_code, state).map_err(|e| {
            warn!(job_id, error = %e, "finish_srun_job: complete_job failed");
            SrunCompleteError::Internal {
                job_id,
                message: e.to_string(),
            }
        })?;
        Ok(job)
    }

    /// Suspend a running job: validate state, record through Raft. Allocation is retained.
    /// The requesting `user` must be the job owner, root, or empty (trusted internal calls).
    pub fn suspend_job(&self, job_id: JobId, user: &str) -> anyhow::Result<()> {
        {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            if job.state != JobState::Running {
                anyhow::bail!("job {} is not running (state {:?})", job_id, job.state);
            }
            Self::check_job_owner(user, &job.spec.user, "suspend")?;
        }
        self.propose(WalOperation::JobSuspend {
            job_id,
            at: chrono::Utc::now(),
            preempted_by: None,
            preempt_qos: None,
        })?;
        info!(job_id, "job suspended");
        Ok(())
    }

    /// Resume a suspended job: validate state, record through Raft, fold suspended time.
    /// The requesting `user` must be the job owner, root, or empty (trusted internal calls).
    pub fn resume_job(&self, job_id: JobId, user: &str) -> anyhow::Result<()> {
        {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            if job.state != JobState::Suspended {
                anyhow::bail!("job {} is not suspended (state {:?})", job_id, job.state);
            }
            Self::check_job_owner(user, &job.spec.user, "resume")?;
        }
        self.propose(WalOperation::JobResume {
            job_id,
            at: chrono::Utc::now(),
        })?;
        info!(job_id, "job resumed");
        Ok(())
    }

    /// Start a job on specific nodes.
    /// Transition a pending job to Running and record its allocation. Returns
    /// the run epoch assigned to this dispatch (threaded into the launch RPC).
    pub fn start_job(
        &self,
        job_id: JobId,
        node_names: Vec<String>,
        resources: ResourceAllocations,
        per_node_alloc: std::collections::HashMap<String, ResourceAllocations>,
    ) -> anyhow::Result<u32> {
        self.start_job_impl(job_id, node_names, resources, per_node_alloc, false)
    }

    pub(crate) fn start_job_impl(
        &self,
        job_id: JobId,
        node_names: Vec<String>,
        resources: ResourceAllocations,
        per_node_alloc: std::collections::HashMap<String, ResourceAllocations>,
        srun_step_dispatch: bool,
    ) -> anyhow::Result<u32> {
        for name in &node_names {
            if !per_node_alloc.contains_key(name) {
                anyhow::bail!(
                    "job {}: per_node_alloc missing entry for node '{}'",
                    job_id,
                    name
                );
            }
        }

        // Validate job exists and can transition
        let old_state;
        let spec_for_notify;
        let submit_time_for_notify;
        let run_attempt;
        {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            old_state = job.state;
            spec_for_notify = job.spec.clone();
            submit_time_for_notify = job.submit_time;
            // Next run epoch (first dispatch = 1), threaded to the agents.
            run_attempt = job.run_attempt.saturating_add(1);
            if job.state != JobState::Pending {
                anyhow::bail!("job {} cannot start from state {:?}", job_id, job.state);
            }
        }

        // propose() handles: state transition, resource allocation, license subtraction
        self.propose(WalOperation::job_state_change(
            job_id,
            old_state,
            JobState::Running,
        ))?;
        self.propose(WalOperation::JobStart {
            job_id,
            nodes: node_names.clone(),
            resources: resources.clone(),
            per_node_alloc: per_node_alloc.clone(),
            srun_step_dispatch,
            run_attempt,
        })?;

        let node_count = node_names.len().max(1) as u32;
        let per_node = node_names
            .first()
            .and_then(|n| per_node_alloc.get(n).cloned())
            .unwrap_or_else(|| {
                ResourceAllocations::with_scalar(
                    resources.cpus / node_count,
                    resources.memory_mb / node_count as u64,
                )
            });
        if !srun_step_dispatch {
            let batch_step = JobStep {
                job_id,
                step_id: STEP_BATCH,
                name: "batch".into(),
                state: StepState::Running,
                num_tasks: 1,
                cpus_per_task: per_node.cpus,
                resources: per_node,
                nodes: node_names,
                distribution: spur_core::step::TaskDistribution::Block,
                start_time: Some(Utc::now()),
                end_time: None,
                exit_code: None,
            };
            if let Err(e) = self.create_step(batch_step) {
                warn!(job_id, error = %e, "failed to record batch step");
            }
        }

        if spec_for_notify
            .mail_type
            .iter()
            .any(|t| t == "BEGIN" || t == "ALL")
        {
            self.send_notification(job_id, "BEGIN", &spec_for_notify);
        }

        if let Some(ref notifier) = *self.accounting.read() {
            notifier.notify_job_start(JobStartRecord {
                job_id,
                name: spec_for_notify.name.clone(),
                user: spec_for_notify.user.clone(),
                account: spec_for_notify.account.clone().unwrap_or_default(),
                partition: spec_for_notify.partition.clone().unwrap_or_default(),
                qos: spec_for_notify.qos.clone().unwrap_or_default(),
                num_nodes: spec_for_notify.num_nodes,
                num_tasks: spec_for_notify.num_tasks,
                cpus_per_task: spec_for_notify.cpus_per_task,
                memory_mb: resources.memory_mb,
                submit_time: submit_time_for_notify,
                start_time: Utc::now(),
                reservation: spec_for_notify.reservation.clone(),
            });
        }

        debug!(job_id, "job started");
        Ok(run_attempt)
    }

    /// Record completion from one allocated node (multi-node COMPLETING flow).
    pub fn node_complete(
        &self,
        job_id: JobId,
        node_name: &str,
        exit_code: i32,
        signal: i32,
        run_attempt: u32,
    ) -> Result<NodeCompleteResult, NodeCompleteError> {
        {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or(NodeCompleteError::JobNotFound { job_id })?;
            if job.state.is_terminal() {
                return Ok(NodeCompleteResult::AlreadyTerminal);
            }
            // Drop a report from a superseded run (older epoch); e.g. the
            // delayed SIGKILL of a preemption-requeued process. Epoch 0 on
            // either side (legacy) disables the check.
            if run_attempt != 0 && job.run_attempt != 0 && run_attempt < job.run_attempt {
                return Ok(NodeCompleteResult::StaleReport);
            }
            if !job.allocated_nodes.iter().any(|n| n == node_name) {
                return Err(NodeCompleteError::NodeNotAllocated {
                    job_id,
                    node: node_name.to_string(),
                });
            }
        }

        let resp = self
            .propose(WalOperation::JobNodeComplete {
                job_id,
                node_name: node_name.to_string(),
                exit_code,
                signal,
            })
            .map_err(|source| NodeCompleteError::RaftPropose { source })?;

        self.run_all_finalized_side_effects(&resp);
        if let Some(f) = resp.jobs_finalized.first() {
            return Ok(NodeCompleteResult::AllDone {
                state: f.state,
                exit_code: f.exit_code,
            });
        }

        let jobs = self.jobs.read();
        if jobs.get(&job_id).is_some_and(|job| job.state.is_terminal()) {
            return Ok(NodeCompleteResult::AlreadyTerminal);
        }

        Ok(NodeCompleteResult::Completing)
    }

    /// Complete a job (controller-initiated or force-finish from COMPLETING timeout).
    pub fn complete_job(
        &self,
        job_id: JobId,
        exit_code: i32,
        state: JobState,
    ) -> anyhow::Result<()> {
        // A time-limit expiry has to win over the caller's outcome here, not
        // just in the per-node completion path. The marker is only ever set on
        // a run the watchdog signalled, so any completion routed through this
        // method — the completing-timeout force-finish and the srun path, both
        // of which finalize without seeing node_completions — is a wall-time
        // expiry that would otherwise be reported as an ordinary failure and
        // skip the Timeout requeue. Cancelled (user cancel, preemption) and an
        // already-correct Timeout pass through untouched.
        let state = {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            if job.state.is_terminal() {
                anyhow::bail!("invalid transition from {:?} to {:?}", job.state, state);
            }
            if job.time_limit_signaled_at.is_some()
                && matches!(state, JobState::Failed | JobState::Completed)
            {
                JobState::Timeout
            } else {
                state
            }
        };

        // propose() handles: state transition, exit_code, end_time,
        // resource deallocation, step completion, license return
        let resp = self.propose(WalOperation::JobComplete {
            job_id,
            exit_code,
            state,
        })?;
        self.run_all_finalized_side_effects(&resp);

        debug!(job_id, exit_code, "job completed");
        Ok(())
    }

    /// Record that a running job has exhausted its time limit, before the
    /// caller sends SIGTERM. Durably marking the run first is what lets the
    /// completion path report `Timeout` instead of reading the terminating
    /// signal as an ordinary failure — a job that exits promptly on SIGTERM
    /// reports back long before the grace period is up.
    pub fn signal_time_limit(&self, job_id: JobId, at: DateTime<Utc>) -> anyhow::Result<()> {
        self.propose(WalOperation::JobTimeLimitSignaled { job_id, at })?;
        Ok(())
    }

    /// Preempt a running job per its partition's PreemptMode, without
    /// recording a triggering job (e.g. controller-initiated fencing rather
    /// than scheduler priority preemption). See `preempt_job_with_provenance`.
    pub fn preempt_job(&self, job_id: JobId, mode: PreemptMode) -> anyhow::Result<PreemptOutcome> {
        self.preempt_job_with_provenance(job_id, mode, None, None)
    }

    /// Preempt a running job per its partition's PreemptMode. Does the
    /// controller-side state change; the caller dispatches the signal named by
    /// the returned `PreemptOutcome`. `Off` is rejected.
    ///
    /// `preempted_by` is the job that triggered the preemption (recorded for
    /// provenance), or `None` when there isn't one. `preempt_qos` is the
    /// authorizing QOS name when `preempt_type = QosPriority`; `None` for
    /// plain priority-based preemption.
    pub fn preempt_job_with_provenance(
        &self,
        job_id: JobId,
        mode: PreemptMode,
        preempted_by: Option<JobId>,
        preempt_qos: Option<String>,
    ) -> anyhow::Result<PreemptOutcome> {
        {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            if job.state != JobState::Running {
                anyhow::bail!("job {} is not running (state {:?})", job_id, job.state);
            }
        }

        match mode {
            PreemptMode::Off => anyhow::bail!("preemption disabled for job {}", job_id),
            PreemptMode::Suspend => {
                // Propose JobSuspend directly (rather than suspend_job) so
                // provenance fields are inside the WAL entry and survive Raft
                // replication and leader failover. The job state was already
                // validated at the top of preempt_job — no re-check needed.
                let resp = self.propose(WalOperation::JobSuspend {
                    job_id,
                    at: chrono::Utc::now(),
                    preempted_by,
                    preempt_qos: preempt_qos.clone(),
                })?;
                self.run_all_finalized_side_effects(&resp);
                info!(job_id, "job preempted (suspend)");
                Ok(PreemptOutcome::Suspended)
            }
            PreemptMode::Cancel => {
                let resp = self.propose(WalOperation::JobPreemptCancel {
                    job_id,
                    preempted_by,
                    preempt_qos,
                })?;
                self.run_all_finalized_side_effects(&resp);
                info!(job_id, "job preempted (cancel)");
                Ok(PreemptOutcome::Killed)
            }
            PreemptMode::Requeue => {
                // Single atomic op: free nodes, end the run for accounting, and
                // return to Pending with an eligibility hold. A two-proposal
                // sequence could strand the job in PREEMPTED if the second
                // proposal failed after the first committed (leadership change /
                // restart), which nothing scans for or recovers.
                //
                // Requeue-by-preemption intentionally ignores spec.requeue and
                // the maybe_requeue MAX_REQUEUE cap: Slurm always requeues a
                // preempted job regardless of its --requeue flag. This is a
                // deliberate divergence from the ordinary requeue path.
                let hold_secs = (self.scheduler_interval_secs as i64 * 2 + 3).max(5);
                let hold = Utc::now() + chrono::Duration::seconds(hold_secs);
                // Honor a later user --begin: compute the max on the leader so
                // followers apply one verbatim instant (no per-replica clock).
                let begin_time = self
                    .jobs
                    .read()
                    .get(&job_id)
                    .and_then(|j| j.spec.begin_time)
                    .map_or(hold, |user_begin| user_begin.max(hold));
                let resp = self.propose(WalOperation::JobPreemptRequeue {
                    job_id,
                    begin_time,
                    preempted_by,
                    preempt_qos,
                })?;
                self.run_all_finalized_side_effects(&resp);
                info!(job_id, hold_secs, "job preempted (requeue)");
                Ok(PreemptOutcome::Killed)
            }
        }
    }

    fn run_job_finalized_side_effects(&self, finalized: JobFinalized) {
        if let Some(stats) = self.sched_stats.get() {
            stats.record_finalized();
            // JobPreemptCancel and JobPreemptRequeue are the only WAL
            // operations that set state=Preempted; Suspend does not.
            if finalized.state == JobState::Preempted {
                stats.record_preempted();
            }
        }
        self.run_epilog_slurmctld(finalized.job_id);
        self.notify_job_finished(finalized.job_id, finalized.state, finalized.exit_code);
        self.react_to_health_job_finalized(finalized.job_id, finalized.state, finalized.exit_code);
    }

    fn run_all_finalized_side_effects(&self, resp: &ClientResponse) {
        for f in &resp.jobs_finalized {
            self.run_job_finalized_side_effects(*f);
        }
    }

    /// One health-check scheduling pass, run each scheduler tick on the leader.
    ///
    /// For each configured check and eligible node it submits the check as an
    /// exclusive whole-node job when it is due. Exclusivity means the scheduler
    /// only starts it once the node is idle, so "probe only when idle" falls out
    /// of normal scheduling. If a check stays queued past `max_wait_secs` because
    /// the node keeps running user jobs, the node is soft-drained so it empties,
    /// then resumed so the pending check can land.
    pub fn run_node_health_pass(&self) {
        let config = self.config();
        if config.health.checks.is_empty() {
            return;
        }
        let now = std::time::Instant::now();
        let now_utc = Utc::now();
        let nodes = self.get_nodes();

        // Bound how many *distinct nodes* are in health activity at once so a
        // restart, failover, or first enable can't set off a fleet-wide check —
        // and, downstream, force-drain — wave. We track a set of node names
        // rather than a plain counter so that multiple checks targeting the same
        // node (e.g. a fast GPU probe + a slow fabric probe) are not counted
        // twice: the node is already exclusively occupied by the first check, so
        // a second check for it doesn't widen the blast radius.
        let cap = config.health.concurrency_cap(nodes.len());
        let mut active_nodes: HashSet<String> = self.health_active_node_names();

        for (idx, check) in config.health.checks.iter().enumerate() {
            // Multi-node checks are rejected at config load (topology-aware
            // pairing is a follow-up); skip defensively without per-tick noise.
            if check.nodes != 1 {
                continue;
            }
            let interval = std::time::Duration::from_secs(check.interval_secs.max(1));
            for node in &nodes {
                let force_draining = matches!(node.state, NodeState::Drain | NodeState::Draining)
                    && node
                        .state_reason
                        .as_deref()
                        .is_some_and(|r| r.starts_with(HEALTH_FORCE_DRAIN_PREFIX));

                // A node we drained to make room for a starved check: resume it
                // the moment it empties so the pending check dispatches. Driven
                // off the durable drain reason, so it holds across a failover.
                if force_draining {
                    if !self.node_has_running_jobs(&node.name) {
                        if let Err(e) =
                            self.update_node_state(&node.name, NodeState::Idle, None, None)
                        {
                            warn!(node = %node.name, error = %e, "failed to resume node for health check");
                        } else {
                            self.scheduler_notify.notify_one();
                        }
                    }
                    continue;
                }

                // Down / failed / operator- or fail-drained: hands off.
                if self.node_off_limits_for_health(node) {
                    continue;
                }

                // An in-flight check already covers this node. If it is stuck
                // pending because the node stays busy, force-drain past max_wait.
                if let Some(existing) = self.pending_health_job(idx, &node.name) {
                    let waited = (now_utc - existing.submit_time).num_seconds();
                    if check.max_wait_secs > 0
                        && existing.state == JobState::Pending
                        && waited >= check.max_wait_secs as i64
                        && self.node_has_running_jobs(&node.name)
                    {
                        let reason = format!(
                            "{HEALTH_FORCE_DRAIN_PREFIX}running overdue check {}",
                            check.program
                        );
                        if let Err(e) = self.drain_node(&node.name, Some(reason), None) {
                            warn!(node = %node.name, error = %e, "failed to drain node for health check");
                        }
                    }
                    continue;
                }

                // No check in flight: submit one if it is due and the health
                // activity budget allows another node.
                let due = self
                    .health_last_check
                    .lock()
                    .get(&(idx, node.name.clone()))
                    .is_none_or(|t| now.duration_since(*t) >= interval);
                if due {
                    // Only count against the cap when this node is not already
                    // in health activity — a second check for the same node
                    // doesn't add a new node to the active set.
                    if cap != 0 && !active_nodes.contains(&node.name) && active_nodes.len() >= cap {
                        continue;
                    }
                    match self.submit_health_job(idx, check, node) {
                        Ok(_) => {
                            active_nodes.insert(node.name.clone());
                        }
                        Err(e) => {
                            warn!(node = %node.name, check = %check.program, error = %e, "failed to submit health-check job")
                        }
                    }
                }
            }
        }
    }

    /// The first non-terminal health-check job for `(idx, node)`, if one is in
    /// flight. Read from the durable job store so it holds across a failover.
    fn pending_health_job(&self, idx: usize, node: &str) -> Option<Job> {
        let name = format!("{HEALTH_JOB_PREFIX}{idx}.{node}");
        self.jobs
            .read()
            .values()
            .find(|j| j.spec.name == name && !j.state.is_terminal())
            .cloned()
    }

    /// How many distinct nodes currently have a non-terminal health-check job
    /// (pending or running). Read from the durable job store so the cap holds
    /// across a failover. A force-drained node still has its pending check, so
    /// bounding this bounds the force-drains too.
    /// Names of nodes that currently have a non-terminal health-check job
    /// (pending or running). Read from the durable job store so the cap holds
    /// across a failover. A force-drained node still has its pending check, so
    /// bounding this set bounds the force-drains too.
    fn health_active_node_names(&self) -> HashSet<String> {
        self.jobs
            .read()
            .values()
            .filter(|j| !j.state.is_terminal())
            .filter_map(|j| health_job_target_node(&j.spec.name).map(str::to_owned))
            .collect()
    }

    /// Whether the health system must leave `node` untouched: only a node that
    /// is up and schedulable (idle/allocated/mixed) is probed. A down node, or
    /// one already drained (by an operator or a failed check), is off-limits —
    /// a failed check's drain is sticky until an operator resumes the node.
    fn node_off_limits_for_health(&self, node: &Node) -> bool {
        !matches!(
            node.state,
            NodeState::Idle | NodeState::Allocated | NodeState::Mixed
        )
    }

    /// Submit one check as an exclusive whole-node job pinned to `node`. It runs
    /// as the check's configured (unprivileged by default) identity, inside the
    /// node's own partition so the pin resolves, and is bounded by `timeout_secs`
    /// so a hung check is killed.
    fn submit_health_job(
        &self,
        idx: usize,
        check: &HealthCheck,
        node: &Node,
    ) -> anyhow::Result<JobId> {
        let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);
        let spec = JobSpec {
            name: format!("{HEALTH_JOB_PREFIX}{idx}.{}", node.name),
            user: check.user.clone(),
            uid: check.uid,
            gid: check.gid,
            partition: node.partitions.first().cloned(),
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            exclusive: true,
            nodelist: Some(node.name.clone()),
            argv: vec![check.program.clone()],
            time_limit: Some(chrono::Duration::seconds(check.timeout_secs.max(1) as i64)),
            ..Default::default()
        };
        // Trusted internal submit: bypass the user-facing validation pipeline
        // (auth, account/QOS, submit limits) that does not apply to a system job.
        self.propose(WalOperation::JobSubmit {
            job_id,
            spec: Box::new(spec),
        })?;
        self.scheduler_notify.notify_one();
        info!(job_id, node = %node.name, check = %check.program, "submitted health-check job");
        Ok(job_id)
    }

    /// React to a finished health-check job: a non-zero exit (or timeout/failure)
    /// drains the node with a sticky, operator-recoverable reason; a pass leaves
    /// the node schedulable. A no-op for any non-health job.
    fn react_to_health_job_finalized(&self, job_id: JobId, state: JobState, exit_code: i32) {
        let Some(job) = self.get_job(job_id) else {
            return;
        };
        let Some(rest) = job.spec.name.strip_prefix(HEALTH_JOB_PREFIX) else {
            return;
        };

        // Record completion so the next check on this node is due one interval
        // from now (best-effort; the store is authoritative for in-flight state).
        if let Some((idx_str, node_from_name)) = rest.split_once('.') {
            if let Ok(idx) = idx_str.parse::<usize>() {
                self.health_last_check
                    .lock()
                    .insert((idx, node_from_name.to_string()), std::time::Instant::now());
            }
        }

        let Some(node) = job.allocated_nodes.first().cloned() else {
            // Never ran (e.g. cancelled while pending): nothing to drain.
            return;
        };
        let program = job.spec.argv.first().cloned().unwrap_or_default();

        // Exit 126/127 from the bash wrapper means the program could not be run
        // (not found / not executable) — a deployment or config error, not a
        // hardware fault. Draining the node for it would drain the whole fleet
        // over one un-distributed script, so log it loudly and leave the node in
        // service instead of taking a sticky hardware drain.
        if state == JobState::Failed && matches!(exit_code, 126 | 127) {
            warn!(
                node = %node,
                program = %program,
                exit_code,
                "health check could not run (missing or non-executable program?); \
                 not draining the node — fix the program path/distribution"
            );
            return;
        }

        let failed = matches!(
            state,
            JobState::Failed
                | JobState::Timeout
                | JobState::NodeFail
                | JobState::Deadline
                | JobState::OutOfMemory
        );

        if failed {
            let reason =
                format!("{HEALTH_FAIL_DRAIN_PREFIX}{program} failed ({state:?}, exit {exit_code})");
            if let Err(e) = self.drain_node(&node, Some(reason), None) {
                warn!(node = %node, error = %e, "failed to drain node after failing health check");
            }
        }
        // Passed/Cancelled/Preempted: leave the node as-is. A force-drained node
        // was already resumed by the pass before the check could run.
    }

    fn run_epilog_slurmctld(&self, job_id: JobId) {
        let Some(epilog_ctld) = self.config().hooks.epilog_slurmctld.clone() else {
            return;
        };
        let job = self.get_job(job_id);
        let ctx = spur_core::hooks::HookContext {
            job_id,
            work_dir: job
                .as_ref()
                .map(|j| j.spec.work_dir.clone())
                .unwrap_or_else(|| "/tmp".into()),
            uid: job.as_ref().map(|j| j.spec.uid).unwrap_or(0),
            gid: job.as_ref().map(|j| j.spec.gid).unwrap_or(0),
            partition: job
                .as_ref()
                .and_then(|j| j.spec.partition.clone())
                .unwrap_or_default(),
            nodelist: job
                .as_ref()
                .map(|j| j.allocated_nodes.join(","))
                .unwrap_or_default(),
            script_context: "epilog_slurmctld".into(),
            gpu_devices: Vec::new(),
            cpus: job.as_ref().map(|j| j.spec.cpus_per_task).unwrap_or(1),
            memory_mb: job
                .as_ref()
                .and_then(|j| j.spec.memory_per_node_mb)
                .unwrap_or(0),
        };
        tokio::spawn(async move {
            if let Err(e) = spur_core::hooks::run_hook(&epilog_ctld, &ctx).await {
                warn!(job_id, error = %e, "EpilogSlurmctld failed");
            }
        });
    }

    fn notify_job_finished(&self, job_id: JobId, state: JobState, exit_code: i32) {
        let spec_for_notify = self.jobs.read().get(&job_id).map(|j| j.spec.clone());
        if let Some(spec) = spec_for_notify {
            let is_success = state == JobState::Completed;
            let is_failure = matches!(
                state,
                JobState::Failed | JobState::Timeout | JobState::NodeFail | JobState::Deadline
            );
            if is_success && spec.mail_type.iter().any(|t| t == "END" || t == "ALL") {
                self.send_notification(job_id, "END", &spec);
            }
            if is_failure && spec.mail_type.iter().any(|t| t == "FAIL" || t == "ALL") {
                self.send_notification(job_id, "FAIL", &spec);
            }
        }

        if let Some(ref notifier) = *self.accounting.read() {
            let (exit_signal, derived_exit_code, preempted_by, preempt_mode, preempt_qos) = self
                .jobs
                .read()
                .get(&job_id)
                .map(|j| {
                    (
                        j.exit_signal,
                        j.derived_exit_code,
                        j.preempted_by,
                        j.preempt_mode.clone(),
                        j.preempt_qos.clone(),
                    )
                })
                .unwrap_or((0, 0, None, None, None));
            notifier.notify_job_end(
                job_id,
                state,
                exit_code,
                Utc::now(),
                exit_signal,
                derived_exit_code,
                preempted_by,
                preempt_mode,
                preempt_qos,
            );
        }

        // Preempted excluded: preempt_job owns its requeue (with hold).
        let should_requeue = matches!(state, JobState::Timeout | JobState::NodeFail);
        if should_requeue {
            if let Err(e) = self.maybe_requeue(job_id) {
                warn!(job_id, error = %e, "failed to requeue job");
            }
        }
    }

    /// Requeue a job if spec.requeue is set and attempt limit not exceeded.
    fn maybe_requeue(&self, job_id: JobId) -> anyhow::Result<()> {
        let max = self.config().controller.max_batch_requeue;
        let (old_state, backoff) = {
            let jobs = self.jobs.read();
            let Some(job) = jobs.get(&job_id) else {
                return Ok(());
            };
            if job.requeue_count >= max {
                if matches!(
                    job.state,
                    JobState::Preempted | JobState::Timeout | JobState::NodeFail
                ) {
                    drop(jobs);
                    return self.hold_job_at_max_requeue(job_id);
                }
                return Ok(());
            }
            if !job.spec.requeue {
                return Ok(());
            }
            // The eviction tagged the cause, so a launch failure gets the same
            // backoff the all-nodes-failed path gets. Without it a job with one
            // broken node in its allocation burns its whole requeue budget in
            // seconds. Node health and timeout requeues keep today's immediate
            // retry: the blocking condition is already gone by then.
            let launch_failed = job.pending_reason == PendingReason::JobLaunchFailure;
            (
                job.state,
                launch_failed.then(|| self.launch_backoff_until(job)),
            )
        };

        // Computed before the proposal so every replica applies one verbatim
        // instant rather than reading its own clock.
        let op = match backoff {
            Some(hold) => WalOperation::job_state_change_backoff_pending(
                job_id,
                old_state,
                PendingReason::JobLaunchFailure,
                hold,
            ),
            None => WalOperation::job_state_change(job_id, old_state, JobState::Pending),
        };
        self.propose(op)?;

        info!(job_id, from = %old_state, hold_until = ?backoff, "job requeued");
        Ok(())
    }

    /// Requeue a job back to Pending after a dispatch failure.
    /// Unlike `maybe_requeue`, this is unconditional and doesn't require
    /// the requeue flag on the spec. Used when the agent rejects a job
    /// (e.g., container image not found) so it can be retried after the
    /// user fixes the issue. (Issue #91)
    pub fn requeue_job(&self, job_id: JobId) -> anyhow::Result<()> {
        self.requeue_after_launch_failure(job_id, false)
    }

    /// Requeue after a dispatch failure, optionally parking the job for an
    /// operator instead of retrying it.
    ///
    /// `hold` is for failures the job carries with it, which would fail the same
    /// way on the next node and drain that one too. Such a job skips the backoff
    /// and the requeue cap, neither of which means anything to a job that will
    /// not retry on its own.
    pub fn requeue_after_launch_failure(&self, job_id: JobId, hold: bool) -> anyhow::Result<()> {
        let (old_state, begin_time) = {
            let jobs = self.jobs.read();
            let Some(job) = jobs.get(&job_id) else {
                return Ok(());
            };
            if job.state.is_terminal() {
                return Ok(());
            }
            if !hold && job.requeue_count >= self.config().controller.max_batch_requeue {
                drop(jobs);
                return self.hold_job_at_max_requeue(job_id);
            }
            // A job that never reached Running has nothing to requeue: Pending ->
            // Failed is not a legal transition and Pending -> Pending applies as a
            // NoOp, so both proposals below would be discarded and the hold lost.
            // Callers that fail before dispatch fall to the next scheduler tick.
            if job.state == JobState::Pending {
                debug!(
                    job_id,
                    "requeue requested for a job that never started; no backoff hold applied"
                );
                return Ok(());
            }
            (job.state, self.launch_backoff_until(job))
        };

        // transition to Failed via JobComplete so node resources,
        // licenses, and steps are properly cleaned up.
        self.propose(WalOperation::JobComplete {
            job_id,
            exit_code: -1,
            state: JobState::Failed,
        })?;

        if hold {
            self.propose(WalOperation::job_state_change_held_pending_desc(
                job_id,
                JobState::Failed,
                PendingReason::Held,
                LAUNCH_FAILURE_HELD_DESC,
            ))?;
            info!(job_id, from = %old_state, "job requeued and held after launch failure");
            return Ok(());
        }

        // Failed → Pending resets allocation fields and makes the job
        // schedulable again, but only once the backoff hold lapses: without it
        // the next scheduler tick re-dispatches to the same node, which for a
        // node-local fault burns the whole requeue budget in seconds.
        self.propose(WalOperation::job_state_change_backoff_pending(
            job_id,
            JobState::Failed,
            PendingReason::JobLaunchFailure,
            begin_time,
        ))?;

        info!(job_id, from = %old_state, hold_until = %begin_time, "job requeued after dispatch failure");
        Ok(())
    }

    /// Back off a job whose batch dispatch confirmation failed for a non-prolog
    /// reason (agent briefly unreachable, no resolved address) before it left
    /// Pending. `requeue_after_launch_failure` can't be reused: its `requeue_count`
    /// bookkeeping is gated on a real transition out of Running, so without this a
    /// flaky node's job would be reassigned to it every tick, forever unbounded.
    pub(crate) fn backoff_pending_job_after_dispatch_failure(
        &self,
        job_id: JobId,
    ) -> anyhow::Result<()> {
        let begin_time = {
            let jobs = self.jobs.read();
            let Some(job) = jobs.get(&job_id) else {
                return Ok(());
            };
            if job.state != JobState::Pending {
                // Moved on already (e.g. cancelled concurrently) between the
                // failed confirmation and this call — nothing to back off.
                return Ok(());
            }
            if job.requeue_count >= self.config().controller.max_batch_requeue {
                drop(jobs);
                return self.hold_job_at_max_requeue(job_id);
            }
            self.launch_backoff_until(job)
        };

        self.propose(WalOperation::JobDispatchBackoff { job_id, begin_time })?;
        info!(job_id, hold_until = %begin_time, "job's batch dispatch failed before it started; backing off");
        Ok(())
    }

    /// Instant until which a job requeued after a launch failure is held. A user
    /// `--begin` further out always wins, so the hold never shortens a
    /// user-supplied constraint. Computed on the leader so every replica applies
    /// one verbatim instant rather than reading its own clock.
    fn launch_backoff_until(&self, job: &Job) -> DateTime<Utc> {
        let config = self.config();
        let hold_secs = launch_backoff_secs(
            config.scheduler.interval_secs,
            config.controller.max_launch_backoff_secs,
            job.requeue_count,
        );
        let hold = Utc::now() + chrono::Duration::seconds(hold_secs as i64);
        job.spec.begin_time.map_or(hold, |user| user.max(hold))
    }

    /// Evict a running job to NodeFail: frees allocations and feeds the
    /// existing auto-requeue path in `notify_job_finished`, same as the
    /// health-check path (`evict_jobs_on_node`) that runs when an entire
    /// node goes Down, but scoped to a single job. Unlike the health-check
    /// path, this does not by itself cancel the job on nodes that *did*
    /// launch it — a caller evicting a job with launched-but-unconfirmed
    /// nodes is responsible for sending the cancel RPC once eviction
    /// succeeds.
    ///
    /// No longer called from `scheduler_loop`: batch dispatch is now
    /// confirmed on every assigned node *before* a job is allowed to become
    /// Running (see `confirm_dispatch_on_nodes`), so a job can no longer
    /// reach Running with only some of its nodes actually launched — this
    /// function's original trigger. Kept as a public primitive, with its
    /// back-off/requeue contract still exercised directly by this module's
    /// tests, for any other caller that needs to evict an already-Running
    /// job (e.g. a future admin-initiated NodeFail).
    #[allow(dead_code)]
    pub fn evict_job(&self, job_id: JobId) -> anyhow::Result<()> {
        self.evict_job_with_detail(job_id, None)
    }

    pub fn evict_job_with_detail(
        &self,
        job_id: JobId,
        detail: Option<String>,
    ) -> anyhow::Result<()> {
        {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            if job.state.is_terminal() {
                return Ok(());
            }
        }
        let resp = self.propose(WalOperation::JobEvict { job_id, detail })?;
        self.run_all_finalized_side_effects(&resp);
        Ok(())
    }

    /// Gets (creating if absent) the per-node lock `register_node` serializes on.
    fn node_registration_lock(&self, name: &str) -> Arc<parking_lot::Mutex<()>> {
        self.node_registration_locks
            .lock()
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone()
    }

    /// Register a node agent. `caller_privileged` gates a labels change on an already-registered
    /// node (see `sync_node_labels`); first registration never needs it.
    #[allow(clippy::too_many_arguments)]
    pub fn register_node(
        &self,
        name: String,
        hostname: String,
        resources: ResourceSet,
        address: String,
        port: u16,
        wg_pubkey: String,
        version: String,
        source: NodeSource,
        labels: HashMap<String, String>,
        caller_privileged: bool,
    ) -> Result<(), RegisterNodeError> {
        let hostname = if hostname.is_empty() {
            name.clone()
        } else {
            hostname
        };

        // Held across decide-then-propose so concurrent registrations of THIS node can't each act on
        // a pre-write read of the other's state and propose conflicting/duplicate WAL ops.
        let node_lock = self.node_registration_lock(&name);
        let _registration = node_lock.lock();

        let action = {
            let nodes = self.nodes.read();
            evaluate_registration(nodes.get(&name), &resources)
        };

        match action {
            RegistrationAction::Skip => {
                debug!(node = %name, "node unchanged, skipping");
                self.sync_node_labels(&name, labels, caller_privileged)?;
                if let Some(existing) = self.get_node(&name) {
                    let needs_update = existing.address.as_deref() != Some(address.as_str())
                        || existing.hostname != hostname
                        || existing.port != port
                        || (!wg_pubkey.is_empty()
                            && existing.wg_pubkey.as_deref() != Some(wg_pubkey.as_str()))
                        || (!version.is_empty()
                            && existing.version.as_deref() != Some(version.as_str()));
                    if needs_update {
                        self.propose(WalOperation::NodeUpdate {
                            name: name.clone(),
                            hostname: hostname.clone(),
                            resources: existing.total_resources.clone(),
                            address,
                            port,
                            wg_pubkey,
                            version,
                            source: source.clone(),
                        })
                        .map_err(|e| RegisterNodeError::Internal(e.to_string()))?;
                        info!(node = %name, "node comm address or metadata updated");
                    }
                    if existing.source != source {
                        if let Some(node) = self.nodes.write().get_mut(&name) {
                            node.source = source;
                        }
                    }
                }
            }
            RegistrationAction::Update => {
                self.propose(WalOperation::NodeUpdate {
                    name: name.clone(),
                    hostname: hostname.clone(),
                    resources,
                    address,
                    port,
                    wg_pubkey,
                    version,
                    source: source.clone(),
                })
                .map_err(|e| RegisterNodeError::Internal(e.to_string()))?;
                self.sync_node_labels(&name, labels, caller_privileged)?;
                if let Some(node) = self.nodes.write().get_mut(&name) {
                    node.source = source;
                }
                info!(node = %name, "node updated (resources changed)");
            }
            RegistrationAction::Register => {
                self.propose(WalOperation::NodeRegister {
                    name: name.clone(),
                    hostname: hostname.clone(),
                    resources,
                    address,
                    port,
                    wg_pubkey,
                    version,
                    labels,
                    source: source.clone(),
                })
                .map_err(|e| RegisterNodeError::Internal(e.to_string()))?;
                if let Some(node) = self.nodes.write().get_mut(&name) {
                    node.source = source;
                    node.agent_start_time = Some(Utc::now());
                }
                info!(node = %name, "node registered");
            }
        }
        Ok(())
    }

    /// Syncs node labels if they differ, rejecting the diff from an unprivileged caller instead of
    /// applying it. Called under `register_node`'s per-node lock.
    fn sync_node_labels(
        &self,
        node_name: &str,
        new_labels: HashMap<String, String>,
        caller_privileged: bool,
    ) -> Result<(), RegisterNodeError> {
        if let Some(existing) = self.get_node(node_name) {
            if existing.labels != new_labels {
                if !caller_privileged {
                    return Err(RegisterNodeError::PermissionDenied(format!(
                        "relabeling node {node_name} on re-registration requires cluster admin"
                    )));
                }
                let remove: Vec<String> = existing
                    .labels
                    .keys()
                    .filter(|k| !new_labels.contains_key(*k))
                    .cloned()
                    .collect();
                self.propose(WalOperation::NodeLabelsUpdate {
                    name: node_name.to_string(),
                    set: new_labels,
                    remove,
                })
                .map_err(|e| RegisterNodeError::Internal(e.to_string()))?;
                info!(node = %node_name, "node labels synced on re-registration");
            }
        }
        Ok(())
    }

    /// Update node heartbeat telemetry (load, memory, timestamp).
    ///
    /// Returns `true` if the node was found, `false` if unknown.
    /// State recovery is handled separately by `check_node_health`, which
    /// detects the fresh `last_heartbeat` and proposes a WAL-backed transition.
    pub fn update_heartbeat(&self, name: &str, cpu_load: u32, free_memory_mb: u64) -> bool {
        let mut nodes = self.nodes.write();
        if let Some(node) = nodes.get_mut(name) {
            node.cpu_load = cpu_load;
            node.free_memory_mb = free_memory_mb;
            node.last_heartbeat = Some(Utc::now());
            true
        } else {
            false
        }
    }

    /// Record a keepalive from an interactive allocation's client. In-memory
    /// soft state, mirroring `update_heartbeat`; the InactiveLimit reaper reads
    /// it to detect an abandoned allocation.
    pub fn record_job_keepalive(&self, job_id: JobId) {
        self.record_job_keepalive_at(job_id, Utc::now());
    }

    fn record_job_keepalive_at(&self, job_id: JobId, at: DateTime<Utc>) {
        self.interactive_last_seen.write().insert(job_id, at);
    }

    /// Clear keepalive last-seen on leadership (re)gain: a follower records
    /// none, so carried-over stale entries would otherwise reap live allocations.
    pub(crate) fn reset_interactive_last_seen(&self) {
        self.interactive_last_seen.write().clear();
    }

    #[cfg(test)]
    pub(crate) fn keepalive_last_seen(&self, job_id: JobId) -> Option<DateTime<Utc>> {
        self.interactive_last_seen.read().get(&job_id).copied()
    }

    /// Prune the keepalive map to the given live interactive allocations and
    /// return those idle past `limit_secs`. A missing entry is seeded to `now`
    /// (so a just-promoted leader or a newly-seen allocation gets a full window
    /// before it can be reaped). `limit_secs == 0` disables reaping: the map is
    /// still pruned, but nothing is returned.
    pub(crate) fn interactive_reap_candidates(
        &self,
        running_ids: &[JobId],
        now: DateTime<Utc>,
        limit_secs: u32,
    ) -> Vec<JobId> {
        let live: HashSet<JobId> = running_ids.iter().copied().collect();
        let mut seen = self.interactive_last_seen.write();
        seen.retain(|id, _| live.contains(id));
        if limit_secs == 0 {
            return Vec::new();
        }
        let limit = chrono::Duration::seconds(i64::from(limit_secs));
        running_ids
            .iter()
            .copied()
            .filter(|id| now - *seen.entry(*id).or_insert(now) > limit)
            .collect()
    }

    /// Update a node's WireGuard mesh public key from a heartbeat when it appears or changes (mesh
    /// came up after registration, or `spur0` was recreated). In-memory like `update_heartbeat` —
    /// the mesh reconcile loop reads live inventory, so this is enough to include the node in
    /// ApplyMesh without a spurd restart. Returns true if the stored key changed.
    pub fn update_node_wg_pubkey(&self, name: &str, pubkey: &str) -> bool {
        if pubkey.is_empty() {
            return false;
        }
        let mut nodes = self.nodes.write();
        if let Some(node) = nodes.get_mut(name) {
            if node.wg_pubkey.as_deref() != Some(pubkey) {
                node.wg_pubkey = Some(pubkey.to_string());
                return true;
            }
        }
        false
    }

    /// Create an admission token and persist via Raft.
    pub fn create_token(
        &self,
        ttl_secs: Option<u32>,
    ) -> anyhow::Result<(spur_core::admission::AdmissionToken, String)> {
        let (token, full_string) = spur_core::admission::generate_token(ttl_secs);
        self.propose(WalOperation::TokenCreate {
            token: token.clone(),
        })?;
        Ok((token, full_string))
    }

    /// List all admission tokens (without secrets).
    pub fn list_tokens(&self) -> Vec<spur_core::admission::AdmissionToken> {
        self.tokens.read().values().cloned().collect()
    }

    /// Revoke an admission token by ID.
    pub fn revoke_token(&self, token_id: &str) -> anyhow::Result<()> {
        if !self.tokens.read().contains_key(token_id) {
            anyhow::bail!("token not found: {}", token_id);
        }
        self.propose(WalOperation::TokenRevoke {
            token_id: token_id.to_string(),
        })?;
        Ok(())
    }

    /// Get a read-only reference to the token store for validation.
    pub fn get_tokens(&self) -> HashMap<String, spur_core::admission::AdmissionToken> {
        self.tokens.read().clone()
    }

    /// Get all nodes.
    pub fn get_nodes(&self) -> Vec<Node> {
        self.nodes.read().values().cloned().collect()
    }

    /// Nodes eligible for new placement this tick: all nodes minus those on a
    /// dispatch cooldown, except a cooling node pinned by a pending job's
    /// `--nodelist` (its only option). Exclusion is resource-agnostic by design.
    pub fn nodes_off_dispatch_cooldown(&self, pending: &[Job]) -> Vec<Node> {
        let cooling = self.nodes_on_dispatch_cooldown();
        if cooling.is_empty() {
            return self.nodes.read().values().cloned().collect();
        }
        let pinned: HashSet<String> = pending
            .iter()
            .filter_map(|j| j.spec.nodelist.as_deref())
            .filter_map(|nl| spur_core::hostlist::expand(nl).ok())
            .flatten()
            .collect();
        self.nodes
            .read()
            .values()
            .filter(|n| !cooling.contains(&n.name) || pinned.contains(&n.name))
            .cloned()
            .collect()
    }

    /// Get a node by name.
    pub fn get_node(&self, name: &str) -> Option<Node> {
        self.nodes.read().get(name).cloned()
    }

    /// Get all partitions.
    pub fn get_partitions(&self) -> Vec<Partition> {
        self.partitions.read().clone()
    }

    /// Hold a job (prevent scheduling).
    pub fn hold_job(&self, job_id: JobId) -> anyhow::Result<()> {
        let old_priority = {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            if job.state != JobState::Pending {
                anyhow::bail!(
                    "can only hold pending jobs (job {} is {:?})",
                    job_id,
                    job.state
                );
            }
            job.priority
        };

        self.propose(WalOperation::JobPriorityChange {
            job_id,
            old_priority,
            new_priority: 0,
            pending_reason: Some(PendingReason::Held),
            pending_reason_desc: None,
            reset_requeue_count: false,
            clear_reservation: false,
        })?;
        info!(job_id, "job held");
        Ok(())
    }

    /// Hold a *Pending* job whose batch dispatch failed to confirm on a node
    /// (prolog rejection with `hold_on_prolog_fail` set). Same end state as
    /// [`Self::hold_job`] (priority 0, `PendingReason::Held`), but carries the
    /// launch-failure description so it reads the same as the pre-fix path
    /// that reached this via a Running→Failed→Held detour: the job here never
    /// actually left Pending, so that detour isn't available.
    pub(crate) fn hold_job_for_launch_failure(
        &self,
        job_id: JobId,
        reason_desc: Option<&str>,
    ) -> anyhow::Result<()> {
        let old_priority = {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            if job.state != JobState::Pending {
                anyhow::bail!(
                    "can only hold pending jobs (job {} is {:?})",
                    job_id,
                    job.state
                );
            }
            job.priority
        };

        self.propose(WalOperation::JobPriorityChange {
            job_id,
            old_priority,
            new_priority: 0,
            pending_reason: Some(PendingReason::Held),
            pending_reason_desc: Some(reason_desc.unwrap_or(LAUNCH_FAILURE_HELD_DESC).to_string()),
            reset_requeue_count: false,
            clear_reservation: false,
        })?;
        info!(job_id, "job held after launch failure");
        Ok(())
    }

    /// Hold a job that exhausted automatic requeues (`JobHoldMaxRequeue`).
    fn hold_job_at_max_requeue(&self, job_id: JobId) -> anyhow::Result<()> {
        let mut state = {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            job.state
        };

        if state == JobState::Running {
            self.propose(WalOperation::JobComplete {
                job_id,
                exit_code: -1,
                state: JobState::Failed,
            })?;
            state = JobState::Failed;
        }

        if matches!(
            state,
            JobState::Preempted | JobState::Timeout | JobState::NodeFail | JobState::Failed
        ) {
            self.propose(WalOperation::job_state_change_held_pending(
                job_id,
                state,
                PendingReason::JobHoldMaxRequeue,
            ))?;
            info!(job_id, "job held at max requeue limit");
            return Ok(());
        }

        if state != JobState::Pending {
            anyhow::bail!(
                "cannot hold job {} at max requeue from state {:?}",
                job_id,
                state
            );
        }

        let needs_hold = self.jobs.read().get(&job_id).is_some_and(|j| {
            j.pending_reason != PendingReason::JobHoldMaxRequeue || j.priority != 0
        });
        if needs_hold {
            let old_priority = self
                .jobs
                .read()
                .get(&job_id)
                .map(|j| j.priority)
                .unwrap_or(0);
            self.propose(WalOperation::JobPriorityChange {
                job_id,
                old_priority,
                new_priority: 0,
                pending_reason: Some(PendingReason::JobHoldMaxRequeue),
                pending_reason_desc: None,
                reset_requeue_count: false,
                clear_reservation: false,
            })?;
        }
        info!(job_id, "job held at max requeue limit");
        Ok(())
    }

    /// Release a held job.
    pub fn release_job(&self, job_id: JobId) -> anyhow::Result<()> {
        let (reset_requeue, clear_reservation, old_priority) = {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            if !job.pending_reason.is_scheduling_hold() {
                anyhow::bail!("job {} is not held", job_id);
            }
            (
                job.pending_reason == PendingReason::JobHoldMaxRequeue,
                job.pending_reason == PendingReason::ReservationDeleted,
                job.priority,
            )
        };

        self.propose(WalOperation::JobPriorityChange {
            job_id,
            old_priority,
            new_priority: DEFAULT_PRIORITY,
            pending_reason: Some(PendingReason::Priority),
            pending_reason_desc: None,
            reset_requeue_count: reset_requeue,
            clear_reservation,
        })?;
        info!(job_id, "job released");
        Ok(())
    }

    /// Update job properties.
    #[allow(clippy::too_many_arguments)]
    pub fn update_job(
        &self,
        job_id: JobId,
        time_limit: Option<chrono::Duration>,
        priority: Option<u32>,
        partition: Option<String>,
        comment: Option<String>,
        account: Option<String>,
        qos: Option<String>,
    ) -> anyhow::Result<()> {
        let (job_user, job_account) = {
            let jobs = self.jobs.read();
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
            (job.spec.user.clone(), job.spec.account.clone())
        };

        // Reject before mutating: an unknown QOS resolves to the limitless
        // default and an empty QOS clears enforcement — both reopen the bypass.
        let qos = match qos {
            Some(q) => {
                let q = q.trim().to_string();
                if q.is_empty() {
                    anyhow::bail!("cannot clear a job's QOS");
                }
                if self.qos_cache.get(&q).is_none() {
                    let hint = if self.qos_cache.is_loaded() {
                        ""
                    } else {
                        QOS_ACCOUNTING_HINT
                    };
                    anyhow::bail!("QOS '{q}' does not exist{hint}");
                }
                // Treat account="" as unset so we authorize against the
                // job's existing account, rather than erroring on a blank
                // value that check_qos_authorized would otherwise reject.
                let effective_account = account
                    .as_deref()
                    .filter(|a| !a.is_empty())
                    .or(job_account.as_deref());
                if let Some(acct) = effective_account {
                    self.association_cache
                        .check_qos_authorized(&job_user, acct, &q)
                        .map_err(anyhow::Error::msg)?;
                }
                Some(q)
            }
            None => None,
        };

        // Re-validate placement, account and wall-time against the resulting spec
        // before mutating, exactly as submit_job does. An owner editing a job's
        // partition/account/time_limit must still satisfy the same partition ACLs,
        // account associations and MaxTime limits — otherwise a post-submit rewrite
        // would slip a job onto nodes or into an account the user is not entitled to
        // (the QOS arm above already re-validates for the same reason). Reject before
        // mutating so a failed edit leaves the job untouched.
        {
            let partitions = self.partitions.read().clone();
            let config = self.config();
            let candidate = {
                let jobs = self.jobs.read();
                let job = jobs
                    .get(&job_id)
                    .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
                let mut spec = job.spec.clone();
                if let Some(tl) = time_limit {
                    spec.time_limit = Some(tl);
                }
                if let Some(part) = partition.as_ref() {
                    spec.partition = Some(part.clone());
                }
                if let Some(acct) = account.as_ref() {
                    spec.account = Some(acct.clone());
                }
                if let Some(q) = qos.as_ref() {
                    spec.qos = Some(q.clone());
                }
                spec
            };
            validate_user_account(&candidate, &self.association_cache, &config.accounting)?;
            self.validate_partition(&candidate, &partitions)?;
            validate_partition_time_limit(
                &candidate,
                config.scheduler.enforce_part_limits,
                &partitions,
            )?;
        }

        if let Some(p) = priority {
            let old = self
                .jobs
                .read()
                .get(&job_id)
                .map(|j| j.priority)
                .unwrap_or(0);
            self.propose(WalOperation::JobPriorityChange {
                job_id,
                old_priority: old,
                new_priority: p,
                pending_reason: None,
                pending_reason_desc: None,
                reset_requeue_count: false,
                clear_reservation: false,
            })?;
        }

        // Non-WAL-tracked fields: update directly
        let mut jobs = self.jobs.write();
        if let Some(job) = jobs.get_mut(&job_id) {
            if let Some(tl) = time_limit {
                job.spec.time_limit = Some(tl);
            }
            if let Some(part) = partition {
                job.spec.partition = Some(part);
            }
            if let Some(c) = comment {
                job.spec.comment = Some(c);
            }
            if let Some(a) = account {
                job.spec.account = Some(a);
            }
            if let Some(q) = qos {
                job.spec.qos = Some(q);
            }
        }
        info!(job_id, "job updated");
        Ok(())
    }

    /// Record the primary agent's resolved output paths for `scontrol`. Display-only
    /// advisory metadata written straight to the in-memory job — not applied via a WAL
    /// op (though it may ride along in a periodic snapshot), so a failover before the
    /// next snapshot falls back to the computed path.
    /// Empty paths stay `None` (a mixed-version agent decodes the fields as "").
    pub fn set_job_output_paths(&self, job_id: JobId, stdout_path: String, stderr_path: String) {
        let mut jobs = self.jobs.write();
        if let Some(job) = jobs.get_mut(&job_id) {
            job.actual_stdout_path = (!stdout_path.is_empty()).then_some(stdout_path);
            job.actual_stderr_path = (!stderr_path.is_empty()).then_some(stderr_path);
        }
    }

    /// Update node state (admin: drain, resume, etc.)
    ///
    /// When draining a node that still has running jobs, the state is set to
    /// `Draining` instead of `Drain`. Once all jobs complete (tracked in
    /// `complete_job`), the node transitions to `Drain`.
    pub fn update_node_state(
        &self,
        name: &str,
        state: NodeState,
        reason: Option<String>,
        reason_uid: Option<u32>,
    ) -> anyhow::Result<()> {
        let (old_state, effective_state) = {
            let nodes = self.nodes.read();
            let node = nodes
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("node {} not found", name))?;
            let old = node.state;
            let requested = old
                .transition(&NodeEvent::AdminSetState(state), node.admin_locked)
                .unwrap_or(state);
            // Drain with active allocations becomes Draining
            let effective = if requested == NodeState::Drain
                && (node.alloc_resources.cpus > 0 || node.alloc_resources.has_devices())
            {
                NodeState::Draining
            } else {
                requested
            };
            (old, effective)
        };

        // Admin-initiated state changes that move into a hold state are
        // locked so auto-recovery won't override the operator's intent.
        // Resuming to Idle clears the lock.
        let admin_locked = effective_state.is_admin_hold();

        let (reason_uid, reason_time) = reason_attribution(&reason, reason_uid);

        self.propose(WalOperation::NodeStateChange {
            name: name.to_string(),
            old_state,
            new_state: effective_state,
            reason,
            admin_locked,
            reason_uid,
            reason_time,
        })?;
        info!(node = %name, old = ?old_state, new = ?effective_state, "node state updated");
        Ok(())
    }

    pub fn update_node_labels(
        &self,
        name: &str,
        set: HashMap<String, String>,
        remove: &[String],
    ) -> anyhow::Result<()> {
        {
            let nodes = self.nodes.read();
            if !nodes.contains_key(name) {
                anyhow::bail!("node {} not found", name);
            }
        }
        self.propose(WalOperation::NodeLabelsUpdate {
            name: name.to_string(),
            set: set.clone(),
            remove: remove.to_vec(),
        })?;
        info!(node = %name, "node labels updated");
        Ok(())
    }

    /// assign a k0s role + allocated mesh IP + pod /24 to a node (replicated via Raft).
    /// Callers never touch `self.nodes`/`self.k0s` directly — that would bypass Raft.
    pub fn assign_node_k0s(
        &self,
        name: &str,
        role: spur_core::k0s::K0sRole,
        mesh_ip: &str,
        pod_cidr: &str,
    ) -> anyhow::Result<()> {
        {
            let nodes = self.nodes.read();
            if !nodes.contains_key(name) {
                anyhow::bail!("node {} not found", name);
            }
        }
        self.propose(WalOperation::NodeK0sAssign {
            name: name.to_string(),
            role,
            mesh_ip: mesh_ip.to_string(),
            pod_cidr: pod_cidr.to_string(),
        })?;
        info!(node = %name, ?role, "node k0s role assigned");
        Ok(())
    }

    /// Clear a node's k0s role + mesh IP + pod /24 (replicated via Raft). Returns the node to
    /// Spur batch scheduling after k0s teardown. Idempotent: a no-op on an already-cleared node.
    pub fn clear_node_k0s(&self, name: &str) -> anyhow::Result<()> {
        {
            let nodes = self.nodes.read();
            if !nodes.contains_key(name) {
                anyhow::bail!("node {} not found", name);
            }
        }
        self.propose(WalOperation::NodeK0sClear {
            name: name.to_string(),
        })?;
        self.k8s_metrics
            .remove_node(&self.config().cluster_name, name);
        info!(node = %name, "node k0s role cleared");
        Ok(())
    }

    /// Record (or clear, with `None`) why a node blocked k0s convergence, replicated via Raft so
    /// `spur k8s status` can surface it. Errors on an unknown node.
    pub fn set_node_k0s_error(&self, name: &str, error: Option<String>) -> anyhow::Result<()> {
        {
            let nodes = self.nodes.read();
            if !nodes.contains_key(name) {
                anyhow::bail!("node {name} not found");
            }
        }
        self.propose(WalOperation::NodeK0sSetError {
            name: name.to_string(),
            error,
        })?;
        Ok(())
    }

    /// set the cluster-wide k0s phase. A `None`/empty control-plane or `member_nodes` leaves the
    /// persisted value untouched; the `Down` phase clears the member scope + control-plane set.
    pub fn set_k0s_phase(
        &self,
        phase: spur_core::k0s::K0sPhase,
        control_plane_node: Option<String>,
        control_plane_nodes: Vec<String>,
        member_nodes: Vec<String>,
        reset_requested: bool,
    ) -> anyhow::Result<()> {
        use spur_core::k0s::K0sPhase;
        // Serialize prior-phase read, commit, and edge accounting so two concurrent callers can't
        // both observe the same prior phase and double-count the transition.
        let _accounting = self.k0s_phase_accounting.lock();
        let from = self.k0s.read().phase;
        self.propose(WalOperation::K0sSetPhase {
            phase,
            control_plane_node,
            control_plane_nodes,
            member_nodes,
            reset_requested,
        })?;
        if from != phase {
            let cluster = self.config().cluster_name.clone();
            self.k8s_metrics
                .record_phase_transition(&cluster, from, phase);
            if phase == K0sPhase::Provisioning {
                self.k8s_metrics.record_provision_attempt(&cluster);
            }
            // Only a provisioning attempt that gives up counts as a provisioning failure; a
            // Ready -> Degraded runtime fault does not.
            if phase == K0sPhase::Degraded && from == K0sPhase::Provisioning {
                self.k8s_metrics.record_provision_failure(&cluster);
            }
        }
        info!(?phase, "k0s cluster phase set");
        Ok(())
    }

    /// Grow a scoped k0s cluster's member set online: union `nodes` into `member_nodes` (idempotent).
    /// The reconcile loop enrolls the newly-in-scope nodes on its next tick. Caller must ensure the
    /// cluster is scoped (non-empty `member_nodes`) — adding to a whole-inventory cluster would
    /// narrow it (see the `cluster_add_nodes` handler guard).
    pub fn add_k0s_member_nodes(&self, nodes: Vec<String>) -> anyhow::Result<()> {
        self.propose(WalOperation::K0sMemberNodesAdd { nodes })?;
        Ok(())
    }

    /// Shrink a scoped k0s cluster's member set online: subtract `nodes` from `member_nodes`
    /// (idempotent). Caller must ensure the cluster stays scoped (non-empty `member_nodes`) — an
    /// empty set means whole-inventory (see the `cluster_remove_nodes` handler guard).
    pub fn remove_k0s_member_nodes(&self, nodes: Vec<String>) -> anyhow::Result<()> {
        self.propose(WalOperation::K0sMemberNodesRemove { nodes })?;
        Ok(())
    }

    /// snapshot of the current cluster-wide k0s state.
    pub fn k0s_state(&self) -> spur_core::k0s::K0sClusterState {
        self.k0s.read().clone()
    }

    /// The long-lived k0s lifecycle/node metric accumulator.
    pub fn k8s_metrics(&self) -> Arc<spur_metrics::K8sMetrics> {
        self.k8s_metrics.clone()
    }

    /// Current k0s cluster gauges. Role counts come from the incrementally-maintained
    /// `k0s_role_counts` cache rather than a full node scan (see its doc comment).
    pub fn k8s_cluster_metrics(&self) -> spur_metrics::K8sClusterMetricsSnapshot {
        let config = self.config();
        let phase = self.k0s.read().phase;
        let (controller, worker, single) = self.k0s_role_counts.snapshot();
        spur_metrics::K8sClusterMetricsSnapshot::from_role_counts(
            config.cluster_name.clone(),
            u64::from(config.cluster.control_plane_replicas),
            phase,
            controller,
            worker,
            single,
        )
    }

    /// Marks stale nodes Down unless `mark_down` is `Suppressed`, and recovers
    /// nodes whose heartbeat resumed. Returns finalized jobs for cancel RPCs.
    pub fn check_node_health(
        &self,
        timeout_secs: u64,
        mark_down: MarkDownPolicy,
    ) -> Vec<JobFinalized> {
        let mut actions = {
            let nodes = self.nodes.read();
            let refs: Vec<&Node> = nodes.values().collect();
            evaluate_node_health(&refs, Utc::now(), timeout_secs)
        };
        if mark_down == MarkDownPolicy::Suppressed {
            let before = actions.len();
            actions.retain(|a| !matches!(a, HealthAction::MarkDown { .. }));
            let deferred = before - actions.len();
            if deferred > 0 {
                info!(
                    nodes = deferred,
                    "deferring node DOWN during leadership grace period"
                );
            }
        }
        self.apply_health_actions(actions)
    }

    fn apply_health_actions(&self, actions: Vec<HealthAction>) -> Vec<JobFinalized> {
        let mut evicted = Vec::new();
        for action in actions {
            match action {
                HealthAction::MarkDown {
                    name,
                    old_state,
                    admin_locked,
                } => {
                    warn!(node = %name, "node marked DOWN (heartbeat timeout)");
                    // An admin hold's reason takes precedence over the liveness
                    // reason it would otherwise be marked with; keep that hold's
                    // original attribution too. Otherwise this is a system
                    // (heartbeat) action attributed to uid 0 at this instant.
                    let held = admin_locked
                        .then(|| self.get_node(&name))
                        .flatten()
                        .filter(|n| n.state_reason.is_some());
                    let (reason, reason_uid, reason_time) = match held {
                        Some(n) => (n.state_reason, n.reason_uid, n.reason_time),
                        None => (Some("Not responding".into()), Some(0), Some(Utc::now())),
                    };
                    match self.propose(WalOperation::NodeStateChange {
                        name: name.clone(),
                        old_state,
                        new_state: NodeState::Down,
                        reason,
                        admin_locked,
                        reason_uid,
                        reason_time,
                    }) {
                        Ok(resp) => {
                            // A node that stopped heartbeating won't refresh its k0s unit gauge, so
                            // reflect the unreachable unit as down rather than leaving a stale 1.
                            self.k8s_metrics
                                .set_node_up(&self.config().cluster_name, &name, false);
                            self.run_all_finalized_side_effects(&resp);
                            evicted.extend(resp.jobs_finalized);
                        }
                        Err(e) => {
                            warn!(error = %e, "failed to propose node DOWN");
                            continue;
                        }
                    }
                }
                HealthAction::Recover { name, old_state } => {
                    let node = self.get_node(&name);
                    let admin_locked = node.as_ref().is_some_and(|n| n.admin_locked);
                    let recovered_state = recovered_node_state(node.as_ref());
                    // An admin hold applied since the action was computed keeps its reason
                    // and attribution; otherwise recovery clears the liveness reason it is
                    // replacing.
                    let (reason, reason_uid, reason_time) = match node {
                        Some(n) if admin_locked => (n.state_reason, n.reason_uid, n.reason_time),
                        _ => (None, None, None),
                    };
                    info!(node = %name, state = ?recovered_state, "node recovered (heartbeat resumed)");
                    if let Err(e) = self.propose(WalOperation::NodeStateChange {
                        name,
                        old_state,
                        new_state: recovered_state,
                        reason,
                        admin_locked,
                        reason_uid,
                        reason_time,
                    }) {
                        warn!(error = %e, "failed to propose node recovery");
                    }
                }
            }
        }
        evicted
    }

    /// Drain a node: stop scheduling new jobs on it. Running jobs finish naturally.
    /// Returns (actual_state, running_job_count).
    pub fn drain_node(
        &self,
        name: &str,
        reason: Option<String>,
        reason_uid: Option<u32>,
    ) -> anyhow::Result<(NodeState, u32)> {
        let (old_state, running_count) = {
            // Lock order is jobs before nodes, matching apply_operation. Taking
            // nodes first deadlocks against a raft apply that already holds jobs
            // and is waiting on nodes: parking_lot queues writers ahead of new
            // readers, so neither side can make progress and the controller
            // wedges. The agent calls this on a launch failure at the same
            // instant the requeue is being applied for that job.
            let jobs = self.jobs.read();
            let nodes = self.nodes.read();
            let node = nodes
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("node '{}' not found", name))?;
            let count = jobs
                .values()
                .filter(|j| {
                    matches!(
                        j.state,
                        JobState::Running | JobState::Completing | JobState::Suspended
                    ) && j.allocated_nodes.iter().any(|n| n == name)
                })
                .count() as u32;
            (node.state, count)
        };
        let target_state = if running_count > 0 {
            NodeState::Draining
        } else {
            NodeState::Drain
        };
        let (reason_uid, reason_time) = reason_attribution(&reason, reason_uid);
        self.propose(WalOperation::NodeStateChange {
            name: name.to_string(),
            old_state,
            new_state: target_state,
            reason,
            admin_locked: true,
            reason_uid,
            reason_time,
        })?;
        info!(node = %name, state = %target_state, "node drain requested");
        Ok((target_state, running_count))
    }

    /// Whether `name` has any job holding an allocation (Running/Completing/Suspended). Shared by
    /// `remove_node` (inventory) and `cluster_remove_nodes` (k0s membership) so both refuse to yank a
    /// busy node without `--force` using the same rule.
    pub fn node_has_running_jobs(&self, name: &str) -> bool {
        let jobs = self.jobs.read();
        jobs.values().any(|j| {
            matches!(
                j.state,
                JobState::Running | JobState::Completing | JobState::Suspended
            ) && j.allocated_nodes.iter().any(|n| n == name)
        })
    }

    /// Remove a node from the cluster. If `force`, evict running jobs first.
    /// Returns finalized jobs from eviction so callers can send cancel RPCs.
    pub fn remove_node(
        &self,
        name: &str,
        force: bool,
        reason: Option<String>,
    ) -> anyhow::Result<Vec<JobFinalized>> {
        {
            let nodes = self.nodes.read();
            if !nodes.contains_key(name) {
                anyhow::bail!("node '{}' not found", name);
            }
        }
        if !force && self.node_has_running_jobs(name) {
            anyhow::bail!(
                "node '{}' has running jobs; use --force to evict them",
                name
            );
        }

        let resp = self.propose(WalOperation::NodeRemove {
            name: name.to_string(),
            reason,
        })?;
        self.k8s_metrics
            .remove_node(&self.config().cluster_name, name);
        self.run_all_finalized_side_effects(&resp);
        Ok(resp.jobs_finalized)
    }

    /// Create a job step durably via Raft.
    pub fn create_step(&self, step: JobStep) -> anyhow::Result<()> {
        let job_id = step.job_id;
        let step_id = step.step_id;
        self.propose(WalOperation::JobStepCreate {
            step: Box::new(step),
        })?;
        debug!(job_id, step_id, "step created");
        Ok(())
    }

    /// Record an srun step's completion via Raft so the step exit code and the
    /// job's running-max DerivedExitCode are durable and replay-consistent.
    #[allow(clippy::result_large_err)]
    pub fn record_step_complete(
        &self,
        job_id: JobId,
        step_id: u32,
        exit_code: i32,
    ) -> anyhow::Result<()> {
        self.propose(WalOperation::JobStepComplete {
            job_id,
            step_id,
            exit_code,
        })?;
        Ok(())
    }

    /// Look up one step without scanning (and cloning) every step in the cluster.
    pub fn get_step(&self, job_id: JobId, step_id: u32) -> Option<JobStep> {
        self.steps.read().get(&(job_id, step_id)).cloned()
    }

    /// Get all steps for a job.
    pub fn get_steps(&self, job_id: JobId) -> Vec<JobStep> {
        self.steps
            .read()
            .iter()
            .filter(|((jid, _), _)| *jid == job_id)
            .map(|(_, step)| step.clone())
            .collect()
    }

    /// Get pending jobs sorted by priority, filtering out blocked jobs.
    /// Recomputes effective priority using QoS, age, and partition tier before sorting.
    pub fn pending_jobs(&self) -> Vec<Job> {
        self.classify_pending_jobs().jobs
    }

    /// Classify pending jobs once, apply pending-reason updates, advance burst-buffer
    /// stage-in for selected candidates, and return the jobs eligible for scheduling.
    pub fn pending_jobs_and_tag_reasons(&self) -> Vec<Job> {
        let classification = self.classify_pending_jobs();
        let evaluated: Vec<JobId> = classification
            .jobs
            .iter()
            .map(|job| job.job_id)
            .chain(classification.reason_updates.iter().map(|(id, _)| *id))
            .collect();
        self.apply_pending_reason_updates(classification.reason_updates);
        self.advance_bb_staging_for(&classification.bb_stage_candidates);
        self.mark_scheduler_evaluated(&evaluated);
        classification.jobs
    }

    /// Record that this cycle considered these jobs, for `LastSchedEval`. Keyed
    /// by id rather than scanning every job, to bound the write lock.
    fn mark_scheduler_evaluated(&self, job_ids: &[JobId]) {
        if job_ids.is_empty() {
            return;
        }
        let now = Utc::now();
        let mut jobs = self.jobs.write();
        for id in job_ids {
            // Re-check under the write lock: the read snapshot was released, so
            // the job may have started since and must not gain a stamp.
            if let Some(job) = jobs.get_mut(id).filter(|j| j.state == JobState::Pending) {
                job.last_sched_eval = Some(now);
            }
        }
    }

    fn classify_pending_jobs(&self) -> PendingJobClassification {
        let jobs = self.jobs.read();
        let now = Utc::now();
        let running_array_counts: HashMap<JobId, u32> = jobs
            .values()
            .filter(|job| job.state == JobState::Running)
            .filter_map(|job| job.spec.array_job_id)
            .fold(HashMap::new(), |mut counts, array_id| {
                *counts.entry(array_id).or_insert(0) += 1;
                counts
            });
        let mut candidates: Vec<PendingJobCandidate> = jobs
            .values()
            .filter(|job| job.state == JobState::Pending)
            .filter(|job| !job.pending_reason.is_scheduling_hold())
            .filter_map(|job| {
                let before_begin_time = job.spec.begin_time.is_some_and(|begin| now < begin);
                if before_begin_time
                    && matches!(
                        job.pending_reason,
                        PendingReason::BeginTime | PendingReason::Preempted
                    )
                {
                    return None;
                }
                Some(PendingJobCandidate {
                    job: job.clone(),
                    scheduling_eligible: !before_begin_time,
                    tag_reason: job.pending_reason != PendingReason::DeadLine,
                })
            })
            .collect();
        let mut reason_updates = Vec::new();

        // Structural blockers retain their precedence before begin-time eligibility;
        // unlike consumables, they do not reserve capacity while the job waits.
        {
            let partitions = self.partitions.read();
            retain_unblocked(&mut candidates, &mut reason_updates, |job| {
                partition_block(job, &partitions)
            });
        }

        let get_job = |id: JobId| -> Option<Job> { jobs.get(&id).cloned() };
        let get_array_tasks = |id: JobId| -> Vec<Job> {
            jobs.values()
                .filter(|j| j.spec.array_job_id == Some(id))
                .cloned()
                .collect()
        };
        let get_jobs_by_name_user = |name: &str, user: &str| -> Vec<Job> {
            jobs.values()
                .filter(|j| j.spec.name == name && j.spec.user == user)
                .cloned()
                .collect()
        };

        retain_unblocked(&mut candidates, &mut reason_updates, |job| {
            if job.spec.dependency.is_empty() {
                return None;
            }
            use spur_core::dependency::{check_dependencies, DependencyResult};
            match check_dependencies(job, &get_job, &get_array_tasks, &get_jobs_by_name_user) {
                DependencyResult::Satisfied => None,
                DependencyResult::Waiting | DependencyResult::Failed => {
                    Some(PendingReason::Dependency)
                }
            }
        });

        let reservations = self.get_reservations();
        retain_unblocked(&mut candidates, &mut reason_updates, |job| {
            reservation_block(job, &reservations, now)
        });

        let partitions = self.partitions.read();
        for candidate in &mut candidates {
            let job = &mut candidate.job;
            let age_minutes = (now - job.submit_time).num_minutes().max(0);
            let partition_tier =
                spur_core::partition::max_priority_tier(job.spec.partition.as_deref(), &partitions);
            let fair_share = self
                .fairshare_cache
                .get(&job.spec.user, job.spec.account.as_deref().unwrap_or(""));
            job.priority =
                compute_effective_priority(job.priority, fair_share, age_minutes, partition_tier);
        }
        drop(partitions);

        candidates.sort_by(|a, b| {
            let a_res = reservation::job_has_active_reservation(&a.job, &reservations, now);
            let b_res = reservation::job_has_active_reservation(&b.job, &reservations, now);
            b_res
                .cmp(&a_res)
                .then(b.job.priority.cmp(&a.job.priority))
                .then(a.job.job_id.cmp(&b.job.job_id))
        });

        let mut array_counts = running_array_counts;
        for candidate in &mut candidates {
            if !candidate.scheduling_eligible {
                continue;
            }
            let (Some(array_id), Some(max)) = (
                candidate.job.spec.array_job_id,
                candidate.job.spec.array_max_concurrent,
            ) else {
                continue;
            };
            let count = array_counts.entry(array_id).or_insert(0);
            if *count >= max {
                record_pending_reason_update(
                    &mut reason_updates,
                    candidate,
                    PendingReason::JobArrayTaskLimit,
                );
                candidate.scheduling_eligible = false;
                continue;
            }
            if candidate.job.pending_reason == PendingReason::JobArrayTaskLimit {
                record_pending_reason_update(&mut reason_updates, candidate, PendingReason::None);
            }
            *count += 1;
        }

        // Eligible-only: an unreadable cap is transient, not structural, so it must
        // not claim a begin-deferred job and report itself in place of BeginTime.
        // Nothing is admitted past the gate either way — ineligible candidates leave
        // the schedulable set below.
        retain_eligible(&mut candidates, &mut reason_updates, |job| {
            match self.accounting_block(job) {
                Some(reason) => GateOutcome::Block(reason),
                None => GateOutcome::Keep,
            }
        });

        // Built after the accounting gate so a held job never gets the limitless
        // default `resolve_qos` warns about.
        let qos_by_job: HashMap<JobId, Qos> = candidates
            .iter()
            .map(|candidate| (candidate.job.job_id, self.resolve_qos(&candidate.job)))
            .collect();

        {
            let mut reserved = PassReservations::default();
            let grp_wall_usage = self.grp_wall_cache.usage();
            let nodes = self.nodes.read();
            retain_eligible(&mut candidates, &mut reason_updates, |job| {
                let account_charge = match account_block_with(
                    job,
                    &self.association_cache,
                    &jobs,
                    &nodes,
                    &reserved,
                ) {
                    Ok(charge) => charge,
                    Err(reason) => return GateOutcome::Block(reason),
                };
                let consumed_wall = job
                    .spec
                    .qos
                    .as_deref()
                    .and_then(|name| consumed_minutes(&grp_wall_usage, name));
                let qos_charge = match qos_block_with(
                    job,
                    &qos_by_job[&job.job_id],
                    &jobs,
                    &nodes,
                    &reserved,
                    consumed_wall,
                ) {
                    Ok(charge) => charge,
                    Err(reason) => return GateOutcome::Block(reason),
                };
                reserved.reserve(job, qos_charge, account_charge);
                GateOutcome::Keep
            });
        }

        {
            let available = self.available_licenses_with(&jobs);
            let mut remaining = available.clone();
            retain_eligible(&mut candidates, &mut reason_updates, |job| {
                if let Some(reason) = license_block(job, &available) {
                    return GateOutcome::Block(reason);
                }
                let req = extract_license_requirements(&job.spec);
                // In-pass contention counts as a license wait, same as a shortage.
                if req
                    .iter()
                    .any(|(lic, n)| remaining.get(lic).copied().unwrap_or(0) < *n)
                {
                    return GateOutcome::Block(PendingReason::Licenses);
                }
                for (lic, n) in &req {
                    if let Some(avail) = remaining.get_mut(lic) {
                        *avail = avail.saturating_sub(*n);
                    }
                }
                GateOutcome::Keep
            });
        }

        let mut bb_stage_candidates = Vec::new();
        {
            let available = self.available_bb_with(&jobs);
            let mut remaining = available;
            retain_eligible(&mut candidates, &mut reason_updates, |job| {
                if job.bb_stage_state == BbStageState::Staging {
                    return GateOutcome::Block(PendingReason::BurstBufferStageIn);
                }
                if let Some(reason) = burst_buffer_block(job, available) {
                    return GateOutcome::Block(reason);
                }
                let req = extract_bb_requirement(&job.spec);
                if req == 0 {
                    return GateOutcome::Keep;
                }
                if job.bb_stage_state == BbStageState::Ready {
                    return GateOutcome::Keep;
                }
                // In-pass contention counts as a BB wait, same as a shortage.
                if req > remaining {
                    return GateOutcome::Block(PendingReason::BurstBufferResources);
                }
                // Capacity reserved and stage-in scheduled, so the job leaves the
                // schedulable set and shows BurstBufferStageIn next cycle.
                remaining = remaining.saturating_sub(req);
                bb_stage_candidates.push(job.job_id);
                GateOutcome::DropUntagged
            });
        }

        // The only tagger for begin-deferred jobs: the blocker checks above pass
        // ineligible candidates through without claiming them. Running after them
        // preserves their precedence, so a structural blocker still outranks the hold.
        candidates.retain(|candidate| {
            if candidate.scheduling_eligible {
                return true;
            }
            if candidate.job.is_begin_held(now) {
                record_pending_reason_update(
                    &mut reason_updates,
                    candidate,
                    PendingReason::BeginTime,
                );
            }
            false
        });

        PendingJobClassification {
            jobs: candidates
                .into_iter()
                .map(|candidate| candidate.job)
                .collect(),
            reason_updates,
            bb_stage_candidates,
        }
    }

    /// Licenses held by jobs actively occupying resources
    /// (Running/Suspended/Completing). Pending and terminal jobs hold none.
    fn licenses_in_use(jobs: &HashMap<JobId, Job>) -> HashMap<String, u64> {
        let mut used: HashMap<String, u64> = HashMap::new();
        for job in jobs.values() {
            if matches!(
                job.state,
                JobState::Running | JobState::Suspended | JobState::Completing
            ) {
                for (lic, n) in extract_license_requirements(&job.spec) {
                    *used.entry(lic).or_insert(0) += n;
                }
            }
        }
        used
    }

    /// Currently-available licenses: configured total minus licenses in use.
    /// Derived from the live job set, so it always reflects config and cannot
    /// drift (no mutable pool). Caller supplies the already-locked jobs map.
    fn available_licenses_with(&self, jobs: &HashMap<JobId, Job>) -> HashMap<String, u64> {
        let total = self.license_pool.read();
        let used = Self::licenses_in_use(jobs);
        total
            .iter()
            .map(|(lic, tot)| {
                (
                    lic.clone(),
                    tot.saturating_sub(used.get(lic).copied().unwrap_or(0)),
                )
            })
            .collect()
    }

    /// Currently-available licenses (locks the job table). See
    /// [`available_licenses_with`](Self::available_licenses_with).
    #[cfg(test)]
    fn available_licenses(&self) -> HashMap<String, u64> {
        let jobs = self.jobs.read();
        self.available_licenses_with(&jobs)
    }

    /// Burst-buffer capacity (GB) reserved by jobs that have entered staging or
    /// are actively occupying resources. A BB job reserves its capacity when it
    /// transitions to `Staging`; it holds the reservation through Ready, Running,
    /// Suspended, and Completing, releasing only when it leaves the active set.
    /// Pending jobs that have not yet staged (`BbStageState::None`) hold nothing.
    fn bb_capacity_in_use(jobs: &HashMap<JobId, Job>) -> u64 {
        let mut used = 0u64;
        for job in jobs.values() {
            let holds = match job.state {
                JobState::Running | JobState::Suspended | JobState::Completing => true,
                JobState::Pending => job.bb_stage_state != BbStageState::None,
                _ => false,
            };
            if holds {
                used = used.saturating_add(extract_bb_requirement(&job.spec));
            }
        }
        used
    }

    /// Currently-free BB capacity (GB): configured total minus capacity reserved
    /// by staging/active jobs. Derived from the live job set so it always tracks
    /// config and cannot drift. Caller supplies the already-locked jobs map.
    fn available_bb_with(&self, jobs: &HashMap<JobId, Job>) -> u64 {
        let total = *self.burst_buffer_total_gb.read();
        spur_core::burst_buffer::free_capacity_gb(total, Self::bb_capacity_in_use(jobs))
    }

    /// Currently-free BB capacity (locks the job table). See
    /// [`available_bb_with`](Self::available_bb_with).
    #[cfg(test)]
    fn available_bb(&self) -> u64 {
        let jobs = self.jobs.read();
        self.available_bb_with(&jobs)
    }

    /// Advance classifier-selected burst-buffer jobs from `None` to `Staging`.
    /// Stage-in itself is performed out-of-band;
    /// [`complete_bb_stage_in`](Self::complete_bb_stage_in) advances `Staging`
    /// to `Ready`.
    ///
    /// NOTE: the actual data movement (the real stage-in) is a follow-up; this
    /// drives the controller-side state machine and the scheduler hold only.
    fn advance_bb_staging_for(&self, candidates: &[JobId]) -> Vec<JobId> {
        let mut started = Vec::new();
        let mut jobs = self.jobs.write();
        let total = *self.burst_buffer_total_gb.read();
        let mut remaining =
            spur_core::burst_buffer::free_capacity_gb(total, Self::bb_capacity_in_use(&jobs));

        for id in candidates {
            let req = jobs
                .get(id)
                .filter(|job| {
                    job.state == JobState::Pending
                        && job.bb_stage_state == BbStageState::None
                        && !job.pending_reason.is_scheduling_hold()
                        && job.pending_reason != PendingReason::DeadLine
                })
                .map(|job| extract_bb_requirement(&job.spec))
                .unwrap_or(0);
            if req == 0 || req > remaining {
                continue;
            }
            if let Some(job) = jobs.get_mut(id) {
                job.bb_stage_state = BbStageState::Staging;
                job.set_pending_reason(PendingReason::BurstBufferStageIn);
                remaining = remaining.saturating_sub(req);
                started.push(*id);
            }
        }
        started
    }

    #[cfg(test)]
    fn advance_bb_staging(&self) -> Vec<JobId> {
        let candidates = self.classify_pending_jobs().bb_stage_candidates;
        self.advance_bb_staging_for(&candidates)
    }

    /// Drive in-flight burst-buffer stage-ins to completion and return the ids
    /// advanced to `Ready`. Leader-only; called once per scheduler cycle.
    ///
    /// FOLLOW-UP SEAM: real stage-in is asynchronous data movement performed by
    /// the node agent, which would call `complete_bb_stage_in()` over a gRPC
    /// report once the bytes land. Until that round-trip exists, the controller
    /// completes staging here so the lifecycle (`None -> Staging -> Ready ->
    /// dispatch`) is end-to-end functional. Replacing this with an agent report
    /// is the only remaining work; the state machine and scheduler hold are real.
    pub fn drive_bb_stage_in(&self) -> Vec<JobId> {
        let staging: Vec<JobId> = {
            let jobs = self.jobs.read();
            jobs.values()
                .filter(|j| {
                    j.state == JobState::Pending && j.bb_stage_state == BbStageState::Staging
                })
                .map(|j| j.job_id)
                .collect()
        };
        staging
            .into_iter()
            .filter(|id| self.complete_bb_stage_in(*id))
            .collect()
    }

    /// Mark a job's burst-buffer stage-in complete (`Staging -> Ready`), making
    /// it dispatchable. Returns true if the job was advanced. The agent-side
    /// data mover calls this once the bytes have landed (follow-up); tests and
    /// the controller drive it directly.
    pub fn complete_bb_stage_in(&self, job_id: JobId) -> bool {
        let mut jobs = self.jobs.write();
        if let Some(job) = jobs.get_mut(&job_id) {
            if job.state == JobState::Pending && job.bb_stage_state == BbStageState::Staging {
                job.bb_stage_state = BbStageState::Ready;
                if job.pending_reason == PendingReason::BurstBufferStageIn {
                    job.set_pending_reason(PendingReason::None);
                }
                self.scheduler_notify.notify_one();
                return true;
            }
        }
        false
    }

    /// Cancel pending jobs whose dependencies can never be satisfied (Slurm's
    /// `DependencyNeverSatisfied`) and tag still-waiting ones with
    /// `PendingReason::Dependency`. Returns the cancelled ids. Leader-only; takes
    /// the write lock `pending_jobs()` cannot. Closes the silent-deadlock gap
    /// where a `Failed` dependency left the job PENDING forever.
    pub fn cancel_unsatisfiable_dependency_jobs(&self) -> Vec<JobId> {
        use spur_core::dependency::{check_dependencies, DependencyResult};
        use spur_core::job::PendingReason;

        // Snapshot under a read lock to evaluate dependencies.
        let (to_cancel, to_wait): (Vec<JobId>, Vec<JobId>) = {
            let jobs = self.jobs.read();
            let get_job = |id: JobId| -> Option<Job> { jobs.get(&id).cloned() };
            let get_array_tasks = |id: JobId| -> Vec<Job> {
                jobs.values()
                    .filter(|j| j.spec.array_job_id == Some(id))
                    .cloned()
                    .collect()
            };
            let get_jobs_by_name_user = |name: &str, user: &str| -> Vec<Job> {
                jobs.values()
                    .filter(|j| j.spec.name == name && j.spec.user == user)
                    .cloned()
                    .collect()
            };

            let mut cancel = Vec::new();
            let mut wait = Vec::new();
            for job in jobs.values() {
                if job.state != JobState::Pending
                    || job.spec.dependency.is_empty()
                    || job.pending_reason.is_scheduling_hold()
                {
                    continue;
                }
                match check_dependencies(job, &get_job, &get_array_tasks, &get_jobs_by_name_user) {
                    DependencyResult::Failed => cancel.push(job.job_id),
                    DependencyResult::Waiting => wait.push(job.job_id),
                    DependencyResult::Satisfied => {}
                }
            }
            (cancel, wait)
        };

        // Tag waiting jobs (write lock).
        if !to_wait.is_empty() {
            let mut jobs = self.jobs.write();
            for id in &to_wait {
                if let Some(j) = jobs.get_mut(id) {
                    // Don't clobber Held, DeadLine or a begin-hold reason —
                    // matches update_pending_reasons().
                    if j.state == JobState::Pending
                        && !j.pending_reason.is_scheduling_hold()
                        && j.pending_reason != PendingReason::DeadLine
                        && !j.reason_explains_begin_hold(Utc::now())
                    {
                        j.set_pending_reason(PendingReason::Dependency);
                    }
                }
            }
        }

        // Finalize unsatisfiable jobs via the WAL so resources/accounting fire.
        let mut cancelled = Vec::new();
        for id in to_cancel {
            // Re-check Pending: the snapshot's read lock was released, so the
            // job may have started concurrently. Running -> Cancelled is a valid
            // WAL transition that would otherwise destroy live work.
            if self.jobs.read().get(&id).map(|j| j.state) != Some(JobState::Pending) {
                continue;
            }
            match self.propose(WalOperation::JobComplete {
                job_id: id,
                exit_code: -1,
                state: JobState::Cancelled,
            }) {
                Ok(resp) => {
                    self.run_all_finalized_side_effects(&resp);
                    info!(job_id = id, "job cancelled: dependency never satisfied");
                    cancelled.push(id);
                }
                Err(e) => {
                    warn!(job_id = id, error = %e, "failed to cancel unsatisfiable-dependency job");
                }
            }
        }
        cancelled
    }

    /// Reclassify pending jobs and apply their pending-reason updates.
    /// Leader-only; enforced by the scheduler-loop caller, not this function
    /// itself (mirrors `cancel_unsatisfiable_dependency_jobs()`).
    #[cfg(test)]
    fn refresh_pending_reasons(&self) {
        let reason_updates = self.classify_pending_jobs().reason_updates;
        self.apply_pending_reason_updates(reason_updates);
    }

    fn apply_pending_reason_updates(&self, reason_updates: Vec<(JobId, PendingReason)>) {
        if reason_updates.is_empty() {
            return;
        }

        let mut jobs = self.jobs.write();
        let now = Utc::now();
        for (id, reason) in reason_updates {
            if let Some(j) = jobs.get_mut(&id) {
                // Re-check under the write lock: the read snapshot was released,
                // so the job may have started or been held/deadlined since.
                if j.state == JobState::Pending
                    && !j.pending_reason.is_scheduling_hold()
                    && j.pending_reason != PendingReason::DeadLine
                    && !j.reason_explains_begin_hold(now)
                {
                    // A clear derived from an older read snapshot must not erase
                    // a different reason written before this lock was acquired.
                    if reason == PendingReason::None
                        && j.pending_reason != PendingReason::JobArrayTaskLimit
                    {
                        continue;
                    }
                    j.set_pending_reason(reason);
                }
            }
        }
    }

    pub fn create_partition(&self, partition: Partition) -> Result<(), PartitionError> {
        if partition.name.is_empty() {
            return Err(PartitionError::invalid("partition name must not be empty"));
        }
        if self
            .partitions
            .read()
            .iter()
            .any(|p| p.name == partition.name)
        {
            return Err(PartitionError::already_exists(format!(
                "partition '{}' already exists",
                partition.name
            )));
        }
        let resp = self
            .propose(WalOperation::PartitionCreate { partition })
            .map_err(|e| PartitionError::raft(e.to_string()))?;
        if !resp.partition_created {
            return Err(PartitionError::already_exists(
                "partition already exists".to_string(),
            ));
        }
        Ok(())
    }

    /// Update fields of an existing partition (persisted via Raft).
    #[allow(clippy::too_many_arguments)]
    pub fn update_partition(
        &self,
        name: &str,
        nodes: Option<String>,
        selector: Option<std::collections::HashMap<String, String>>,
        state: Option<String>,
        is_default: Option<bool>,
        max_time: Option<String>,
        default_time: Option<String>,
        max_nodes: Option<u32>,
        clear_max_nodes: bool,
        min_nodes: Option<u32>,
        allow_accounts: Option<Vec<String>>,
        allow_groups: Option<Vec<String>>,
        deny_accounts: Option<Vec<String>>,
        deny_qos: Option<Vec<String>>,
        allow_qos: Option<Vec<String>>,
        priority_tier: Option<u32>,
        preempt_mode: Option<String>,
        preempt_exempt_time: Option<Option<u32>>,
    ) -> Result<(), PartitionError> {
        if !self.partitions.read().iter().any(|p| p.name == name) {
            return Err(PartitionError::not_found(format!(
                "partition '{}' not found",
                name
            )));
        }

        let max_time_minutes = if let Some(ref t) = max_time {
            if t.eq_ignore_ascii_case("INFINITE") || t.eq_ignore_ascii_case("UNLIMITED") {
                Some(None)
            } else {
                let m = spur_core::config::parse_time_minutes(t)
                    .ok_or_else(|| PartitionError::invalid(format!("invalid time: {}", t)))?;
                Some(Some(m))
            }
        } else {
            None
        };

        let default_time_minutes = if let Some(ref t) = default_time {
            if t.eq_ignore_ascii_case("INFINITE") || t.eq_ignore_ascii_case("UNLIMITED") {
                Some(None)
            } else {
                let m = spur_core::config::parse_time_minutes(t).ok_or_else(|| {
                    PartitionError::invalid(format!("invalid default_time: {}", t))
                })?;
                Some(Some(m))
            }
        } else {
            None
        };

        let max_nodes_wal = if clear_max_nodes {
            Some(None)
        } else {
            max_nodes.map(Some)
        };

        self.propose(WalOperation::PartitionUpdate {
            name: name.to_string(),
            nodes,
            selector,
            state,
            max_time_minutes,
            default_time_minutes,
            max_nodes: max_nodes_wal,
            min_nodes,
            allow_accounts,
            allow_groups,
            deny_accounts,
            deny_qos,
            allow_qos,
            priority_tier,
            preempt_mode,
            is_default,
            preempt_exempt_time,
        })
        .map_err(|e| PartitionError::raft(e.to_string()))?;
        Ok(())
    }

    /// Delete a partition by name (persisted via Raft).
    ///
    /// Refuses if any running jobs are using the partition.
    pub fn delete_partition(&self, name: &str) -> Result<(), PartitionError> {
        if !self.partitions.read().iter().any(|p| p.name == name) {
            return Err(PartitionError::not_found(format!(
                "partition '{}' not found",
                name
            )));
        }
        for job in self.jobs.read().values() {
            if !matches!(
                job.state,
                JobState::Running | JobState::Completing | JobState::Suspended
            ) {
                continue;
            }
            if job.spec.partition.as_deref() == Some(name) {
                return Err(PartitionError::invalid(format!(
                    "partition '{}' in use by running job {}",
                    name, job.job_id
                )));
            }
        }
        self.propose(WalOperation::PartitionDelete {
            name: name.to_string(),
        })
        .map_err(|e| PartitionError::raft(e.to_string()))?;
        Ok(())
    }

    /// Re-read spur.conf and apply it to the running controller. Leader-only, and
    /// the swap is in-memory here — no WAL entry carries it, so followers keep
    /// their startup config. Per-field scope: docs/admin-guide/configuration.rst,
    /// "Applying configuration changes" section.
    pub fn reconfigure(&self) -> Result<(), anyhow::Error> {
        let Some(ref path) = self.config_path else {
            anyhow::bail!("reconfigure requires a config file path, but none is configured");
        };
        let new_config = spur_core::config::SlurmConfig::load_from_file(path)?;
        // Reject a broken submit hook before it goes live, so reconfigure can't
        // silently swap in a hook that fails every subsequent submission.
        crate::hooks::validate_submit_hooks(&new_config.hooks)?;
        let conf_partitions = new_config.build_partitions();

        let conf_names: std::collections::HashSet<String> =
            conf_partitions.iter().map(|p| p.name.clone()).collect();
        let current_names: std::collections::HashSet<String> = self
            .partitions
            .read()
            .iter()
            .map(|p| p.name.clone())
            .collect();

        // Delete partitions absent from the new conf.
        for name in current_names.difference(&conf_names) {
            // Skip if active jobs are running on the partition — callers should
            // drain first, matching Slurm's behaviour.
            let in_use = self.jobs.read().values().any(|j| {
                matches!(
                    j.state,
                    JobState::Running | JobState::Completing | JobState::Suspended
                ) && j.spec.partition.as_deref() == Some(name.as_str())
            });
            if in_use {
                warn!(
                    partition = %name,
                    "reconfigure: partition removed from conf but has active jobs; skipping deletion"
                );
                continue;
            }
            self.propose(WalOperation::PartitionDelete { name: name.clone() })
                .map_err(|e| anyhow::anyhow!("reconfigure: delete {name}: {e}"))?;
        }

        // Create partitions present in conf but absent from current WAL state.
        // `WalOperation::PartitionCreate`'s own apply already clears any
        // tombstone for the name being (re)created — no separate step needed.
        for part in conf_partitions
            .iter()
            .filter(|p| !current_names.contains(&p.name))
        {
            self.propose(WalOperation::PartitionCreate {
                partition: part.clone(),
            })
            .map_err(|e| anyhow::anyhow!("reconfigure: create {}: {e}", part.name))?;
        }

        // Update partitions present in both — conf wins unconditionally.
        for part in conf_partitions
            .iter()
            .filter(|p| current_names.contains(&p.name))
        {
            let preempt_str = match part.preempt_mode {
                spur_core::partition::PreemptMode::Cancel => "cancel",
                spur_core::partition::PreemptMode::Requeue => "requeue",
                spur_core::partition::PreemptMode::Suspend => "suspend",
                spur_core::partition::PreemptMode::Off => "off",
            };
            self.propose(WalOperation::PartitionUpdate {
                name: part.name.clone(),
                nodes: Some(part.nodes.clone()),
                selector: Some(part.selector.clone()),
                state: Some(part.state.to_string()),
                max_time_minutes: Some(part.max_time_minutes),
                default_time_minutes: Some(part.default_time_minutes),
                max_nodes: Some(part.max_nodes),
                min_nodes: Some(part.min_nodes),
                allow_accounts: Some(part.allow_accounts.clone()),
                allow_groups: Some(part.allow_groups.clone()),
                deny_accounts: Some(part.deny_accounts.clone()),
                deny_qos: Some(part.deny_qos.clone()),
                allow_qos: Some(part.allow_qos.clone()),
                priority_tier: Some(part.priority_tier),
                preempt_mode: Some(preempt_str.to_string()),
                is_default: Some(part.is_default),
                // Only push a WAL change when TOML explicitly sets the field;
                // absent means "leave whatever the runtime WAL already has".
                preempt_exempt_time: part.preempt_exempt_time.map(Some),
            })
            .map_err(|e| anyhow::anyhow!("reconfigure: update {}: {e}", part.name))?;
        }

        // Re-derive the config-total pools. Availability is computed as total
        // minus in-use elsewhere, so swapping the totals cannot strand
        // capacity already held by running jobs.
        *self.license_pool.write() = new_config.licenses.clone();
        *self.burst_buffer_total_gb.write() = new_config.burst_buffer.total_gb;

        // Swap the live config last, after partition WAL ops have been accepted,
        // so a mid-reconfigure failure leaves the previous config in place.
        // Readers pick up the new sections on their next `config()` snapshot.
        *self.config.write() = Arc::new(new_config);
        // Agent channels are built per call, so new tunables apply to the next RPC. Unlike the
        // jwt key, these carry no security posture, so adopting them live is safe.
        crate::agent_client::set_channel_tuning(&self.config.read().controller);

        // Re-derive per-node `[[nodes]]` policy (features/weight) and partition
        // membership against the freshly-swapped config. This is a local derived
        // projection (recomputed identically on snapshot restore and inside the
        // partition WAL apply handlers), so recomputing it here is consistent.
        // Partition ops above already trigger the same recompute on every peer
        // via the WAL; this covers a nodes-only edit that proposes no op.
        {
            let mut nodes = self.nodes.write();
            self.reconcile_partitions(&mut nodes);
        }

        info!("reconfigure: applied spur.conf on the leader (followers converge on restart; see docs/admin-guide/configuration.rst, \"Applying configuration changes\", for per-field reload scope)");
        Ok(())
    }

    /// Create a new reservation (validated, persisted via Raft).
    pub fn create_reservation(&self, mut res: Reservation) -> Result<(), ReservationError> {
        if res.end_time <= res.start_time {
            return Err(ReservationError::invalid(
                "reservation duration must be positive",
            ));
        }
        if self.reservations.read().iter().any(|r| r.name == res.name) {
            return Err(ReservationError::already_exists(format!(
                "reservation '{}' already exists",
                res.name
            )));
        }
        let known: std::collections::HashSet<String> = self.nodes.read().keys().cloned().collect();
        res.nodes = normalize_node_list(&res.nodes, &known).map_err(ReservationError::invalid)?;
        self.validate_reservation_job_overlap(&res, None)
            .map_err(|e| ReservationError::invalid(e.to_string()))?;
        self.validate_reservation_storage_overlap(&res, None)
            .map_err(|e| ReservationError::invalid(e.to_string()))?;
        let name = res.name.clone();
        let resp = self
            .propose(WalOperation::ReservationCreate { reservation: res })
            .map_err(|e| ReservationError::raft(e.to_string()))?;
        if !resp.reservation_created {
            return Err(ReservationError::already_exists(format!(
                "reservation '{}' already exists",
                name
            )));
        }
        Ok(())
    }

    /// Update an existing reservation (validated, persisted via Raft). Authorization is the RPC
    /// boundary's (`require_reservation_manager`): an operator may manage any reservation.
    #[allow(clippy::too_many_arguments)]
    pub fn update_reservation(
        &self,
        name: &str,
        duration_minutes: u32,
        add_nodes: &[String],
        remove_nodes: &[String],
        add_users: &[String],
        remove_users: &[String],
        add_accounts: &[String],
        remove_accounts: &[String],
    ) -> Result<(), ReservationError> {
        let mut preview = self
            .reservations
            .read()
            .iter()
            .find(|r| r.name == name)
            .cloned()
            .ok_or_else(|| {
                ReservationError::not_found(format!("reservation '{}' not found", name))
            })?;

        if duration_minutes > 0 {
            preview.end_time =
                preview.start_time + chrono::Duration::minutes(duration_minutes as i64);
        }
        // Unreachable today (start_time is immutable and create rejects zero
        // spans), but guard explicitly so a future editable start_time can't
        // silently reintroduce a zero-length, instantly-purged reservation.
        if preview.end_time <= preview.start_time {
            return Err(ReservationError::invalid(
                "reservation duration must be positive",
            ));
        }
        let known: std::collections::HashSet<String> = self.nodes.read().keys().cloned().collect();
        let mut add_expanded = Vec::new();
        for n in add_nodes {
            add_expanded.extend(
                normalize_node_list(std::slice::from_ref(n), &known)
                    .map_err(ReservationError::invalid)?,
            );
        }
        for node in &add_expanded {
            if !preview.nodes.contains(node) {
                preview.nodes.push(node.clone());
            }
        }
        preview.nodes.retain(|n| !remove_nodes.contains(n));
        for add_user in add_users {
            if !preview.users.contains(add_user) {
                preview.users.push(add_user.clone());
            }
        }
        preview.users.retain(|u| !remove_users.contains(u));
        for account in add_accounts {
            if !preview.accounts.contains(account) {
                preview.accounts.push(account.clone());
            }
        }
        preview.accounts.retain(|a| !remove_accounts.contains(a));

        self.validate_reservation_job_overlap(&preview, Some(name))
            .map_err(|e| ReservationError::invalid(e.to_string()))?;
        self.validate_reservation_storage_overlap(&preview, Some(name))
            .map_err(|e| ReservationError::invalid(e.to_string()))?;

        self.propose(WalOperation::ReservationUpdate {
            name: name.to_string(),
            duration_minutes,
            add_nodes: add_expanded,
            remove_nodes: remove_nodes.to_vec(),
            add_users: add_users.to_vec(),
            remove_users: remove_users.to_vec(),
            add_accounts: add_accounts.to_vec(),
            remove_accounts: remove_accounts.to_vec(),
        })
        .map_err(|e| ReservationError::raft(e.to_string()))?;
        Ok(())
    }

    /// Delete a reservation by name (persisted via Raft). Authorization is the RPC boundary's
    /// (`require_reservation_manager`): an operator may manage any reservation.
    pub fn delete_reservation(&self, name: &str) -> Result<(), ReservationError> {
        if !self.reservations.read().iter().any(|r| r.name == name) {
            return Err(ReservationError::not_found(format!(
                "reservation '{}' not found",
                name
            )));
        }

        for job in self.jobs.read().values() {
            if !matches!(
                job.state,
                JobState::Running | JobState::Completing | JobState::Suspended
            ) {
                continue;
            }
            if job.spec.reservation.as_deref() == Some(name) {
                return Err(ReservationError::invalid(format!(
                    "reservation '{}' in use by running job {}",
                    name, job.job_id
                )));
            }
        }

        self.propose(WalOperation::ReservationDelete {
            name: name.to_string(),
        })
        .map_err(|e| ReservationError::raft(e.to_string()))?;
        Ok(())
    }

    /// Fire a best-effort audit write when accounting is configured; no-op
    /// otherwise. Mirrors the job start/end notification path.
    pub(crate) fn record_txn(&self, record: TxnRecord) {
        if let Some(ref notifier) = *self.accounting.read() {
            notifier.notify_txn(record);
        }
    }

    /// Remove reservations past their end time when no jobs still reference them.
    pub fn purge_expired_reservations(&self) {
        let now = Utc::now();
        let expired: Vec<String> = self
            .reservations
            .read()
            .iter()
            .filter(|r| r.is_expired(now))
            .map(|r| r.name.clone())
            .collect();
        for name in expired {
            let in_use = self.jobs.read().values().any(|job| {
                matches!(
                    job.state,
                    JobState::Running | JobState::Completing | JobState::Suspended
                ) && job.spec.reservation.as_deref() == Some(name.as_str())
            });
            if in_use {
                continue;
            }
            let error = match self.propose(WalOperation::ReservationDelete { name: name.clone() }) {
                Ok(_) => None,
                Err(e) => {
                    warn!(name = %name, error = %e, "failed to purge expired reservation");
                    Some(e.to_string())
                }
            };
            // Audit both outcomes (like the RPC handlers) so a repeatedly-failing
            // system purge leaves a trail, not just an ephemeral warn.
            let outcome = if error.is_none() {
                TxnOutcome::Success
            } else {
                TxnOutcome::Error
            };
            self.record_txn(TxnRecord {
                ts: Utc::now(),
                actor: "system".to_string(),
                actor_uid: None,
                verified: false,
                source: TxnSource::System,
                action: TxnAction::Delete,
                entity_type: TxnEntity::Reservation,
                entity_name: name.clone(),
                outcome,
                details: txn::finalize_details(
                    txn::delete_details(Some("expired")),
                    error.as_deref(),
                ),
            });
        }
    }

    /// Evict finished jobs whose end_time is older than the retention window,
    /// bounding controller memory. No-op when nothing has aged out.
    pub fn evict_expired_terminal_jobs(&self) {
        // Floor above the reconcile interval so a job survives at least one
        // accounting reconcile pass before it can be evicted (retention 0 included).
        let retention = self
            .config()
            .controller
            .terminal_job_retention_secs
            .max(crate::accounting::RECONCILE_INTERVAL_SECS);
        let before = Utc::now() - chrono::Duration::seconds(retention as i64);
        let jobs = self.jobs.read();
        // Spare a target still referenced by a live job's dependency: dropping it
        // makes resolve_target_state return None, which cancels/early-releases dependents.
        let mut referenced: HashSet<JobId> = HashSet::new();
        for j in jobs.values().filter(|j| !j.state.is_finalized()) {
            for dep in spur_core::dependency::parse_dependencies(&j.spec.dependency) {
                if let Some(id) = dep.target_job_id() {
                    referenced.insert(id);
                }
            }
        }
        // is_finalized is load-bearing, not redundant: end_time survives a
        // requeue, so a re-dispatched job's stale end_time alone would reap it.
        let job_ids: Vec<JobId> = jobs
            .iter()
            .filter(|(id, j)| {
                j.state.is_finalized()
                    && j.end_time.is_some_and(|t| t < before)
                    && !referenced.contains(id)
                    && j.spec.array_job_id.is_none_or(|p| !referenced.contains(&p))
            })
            .map(|(&id, _)| id)
            .collect();
        drop(jobs);
        if job_ids.is_empty() {
            return;
        }
        if let Err(e) = self.propose(WalOperation::EvictTerminalJobs { job_ids }) {
            warn!(error = %e, "failed to evict expired terminal jobs");
        }
    }

    /// Cancel running jobs whose reservation window has ended (after optional grace).
    pub fn enforce_reservation_end_times(&self) {
        let now = Utc::now();
        let grace = chrono::Duration::minutes(self.config().scheduler.resv_overrun_minutes as i64);
        let reservations: std::collections::HashMap<String, Reservation> = self
            .get_reservations()
            .into_iter()
            .map(|r| (r.name.clone(), r))
            .collect();
        let to_cancel: Vec<JobId> = self
            .jobs
            .read()
            .values()
            .filter_map(|job| {
                if !matches!(
                    job.state,
                    JobState::Running | JobState::Completing | JobState::Suspended
                ) {
                    return None;
                }
                let res_name = job.spec.reservation.as_ref()?;
                let res = reservations.get(res_name)?;
                if now > res.end_time + grace {
                    Some(job.job_id)
                } else {
                    None
                }
            })
            .collect();
        for job_id in to_cancel {
            if let Err(e) = self.complete_job(job_id, -1, JobState::Cancelled) {
                warn!(job_id, error = %e, "failed to cancel job after reservation ended");
            }
        }
    }

    fn validate_reservation_job_overlap(
        &self,
        res: &Reservation,
        except_name: Option<&str>,
    ) -> anyhow::Result<()> {
        if res.flags.ignore_jobs {
            return Ok(());
        }
        let jobs = self.jobs.read();
        if let Some((job_id, node)) =
            running_jobs_overlap_start(&jobs, &res.nodes, res.start_time, except_name)
        {
            anyhow::bail!(
                "requested nodes are busy (job {} on {} until after reservation start)",
                job_id,
                node
            );
        }
        Ok(())
    }

    fn validate_reservation_storage_overlap(
        &self,
        res: &Reservation,
        except_name: Option<&str>,
    ) -> anyhow::Result<()> {
        for existing in self.reservations.read().iter() {
            if except_name == Some(existing.name.as_str()) {
                continue;
            }
            if !reservation::reservations_overlap(res, existing) {
                continue;
            }
            if reservation::overlap_allowed(res, existing) {
                continue;
            }
            anyhow::bail!(
                "reservation overlaps with existing reservation '{}'",
                existing.name
            );
        }
        Ok(())
    }

    fn hold_jobs_for_deleted_reservation_jobs(jobs: &mut HashMap<JobId, Job>, name: &str) {
        for job in jobs.values_mut() {
            if job.state != JobState::Pending {
                continue;
            }
            if job.spec.reservation.as_deref() != Some(name) {
                continue;
            }
            if job.pending_reason.is_scheduling_hold() {
                continue;
            }
            job.priority = 0;
            job.set_pending_reason(PendingReason::ReservationDeleted);
        }
    }

    fn detach_jobs_from_deleted_reservation_jobs(jobs: &mut HashMap<JobId, Job>, name: &str) {
        for job in jobs.values_mut() {
            if job.state != JobState::Pending {
                continue;
            }
            if job.spec.reservation.as_deref() != Some(name) {
                continue;
            }
            job.spec.reservation = None;
            if job.pending_reason == PendingReason::ReservationDeleted {
                job.set_pending_reason(PendingReason::None);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_reservation_update_locked(
        reservations: &mut [Reservation],
        name: &str,
        duration_minutes: u32,
        add_nodes: &[String],
        remove_nodes: &[String],
        add_users: &[String],
        remove_users: &[String],
        add_accounts: &[String],
        remove_accounts: &[String],
    ) {
        let Some(res) = reservations.iter_mut().find(|r| r.name == name) else {
            return;
        };
        if duration_minutes > 0 {
            res.end_time = res.start_time + chrono::Duration::minutes(duration_minutes as i64);
        }
        for node in add_nodes {
            if !res.nodes.contains(node) {
                res.nodes.push(node.clone());
            }
        }
        res.nodes.retain(|n| !remove_nodes.contains(n));
        for user in add_users {
            if !res.users.contains(user) {
                res.users.push(user.clone());
            }
        }
        res.users.retain(|u| !remove_users.contains(u));
        for account in add_accounts {
            if !res.accounts.contains(account) {
                res.accounts.push(account.clone());
            }
        }
        res.accounts.retain(|a| !remove_accounts.contains(a));
    }

    /// Get all reservations.
    pub fn get_reservations(&self) -> Vec<Reservation> {
        self.reservations.read().clone()
    }

    /// Update pending_reason for jobs the scheduler couldn't schedule.
    ///
    /// Called after each scheduling cycle so that `squeue` shows a meaningful
    /// reason instead of always displaying "Priority".
    ///
    /// - `Resources`: no suitable node exists for the job right now
    ///   (partition mismatch, full, constraint not met, etc.)
    /// - `Priority`: suitable nodes exist but they're reserved for
    ///   higher-priority jobs (backfill timeline is in the future)
    /// - `NodeDown`: all nodes in the target partition are down/drained
    pub fn update_pending_reasons(
        &self,
        unscheduled: &[&spur_core::job::Job],
        cluster_state: &spur_sched::traits::ClusterState,
    ) {
        use spur_core::job::PendingReason;

        let mut jobs = self.jobs.write();

        for job in unscheduled {
            let job_entry = match jobs.get_mut(&job.job_id) {
                Some(j) => j,
                None => continue,
            };

            // Don't overwrite held jobs
            if job_entry.pending_reason.is_scheduling_hold() {
                continue;
            }
            // Don't overwrite a DeadLine reason set by the deadline-enforcement
            // path — the job is about to transition to JobState::Deadline this
            // tick; clobbering with Resources/NodeDown would mislead any
            // observer that polls in between.
            if job_entry.pending_reason == PendingReason::DeadLine {
                continue;
            }
            // Keep a reason that explains an active begin_time hold (preemption
            // requeue, launch-failure backoff) until it lapses, then fall through
            // to the real wait reason.
            if job_entry.reason_explains_begin_hold(Utc::now()) {
                continue;
            }

            if let Some(reason) = reservation_fence_reason(job, cluster_state) {
                job_entry.set_pending_reason(reason);
                continue;
            }

            // Reuse the scheduler's matcher so the reason can't disagree with
            // what backfill actually does.
            let placement = spur_sched::node_match::NodePlacement::new(job);
            let now = chrono::Utc::now();

            let needed = (job.spec.num_nodes as usize).max(1);

            let eligible: Vec<&spur_core::node::Node> = cluster_state
                .nodes
                .iter()
                .filter(|n| placement.eligible(n, cluster_state.reservations, now))
                .collect();

            let required = spur_sched::backfill::job_resource_request(job);
            if placement.nodelist_is_additive()
                && eligible.iter().any(|node| {
                    placement.is_listed(&node.name)
                        && node.total_resources.can_satisfy(&required)
                        && !placement.matches_for_reservation(node, cluster_state.reservations, now)
                })
            {
                job_entry.set_pending_reason(PendingReason::ReqNodeNotAvail);
                continue;
            }

            // Fewer eligible nodes than requested: unschedulable as written.
            if eligible.len() < needed {
                let partition_size = cluster_state
                    .nodes
                    .iter()
                    .filter(|n| placement.in_partition(n))
                    .count();

                job_entry.set_pending_reason(if needed > partition_size {
                    PendingReason::PartitionNodeLimit
                } else if job.spec.constraint.is_some() && eligible.is_empty() {
                    PendingReason::BadConstraints
                } else if job.spec.nodelist.as_deref().is_some_and(|s| !s.is_empty())
                    && !placement.nodelist_is_additive()
                {
                    PendingReason::ReqNodeNotAvail
                } else {
                    PendingReason::Resources
                });
                continue;
            }

            // Eligible nodes exist but none are up. is_up() keeps a busy
            // `Allocated` cluster out of NodeDown; a nodelist pin to down nodes
            // is ReqNodeNotAvail (Slurm parity).
            if eligible.iter().all(|n| !n.state.is_up()) {
                job_entry.set_pending_reason(
                    if job.spec.nodelist.as_deref().is_some_and(|s| !s.is_empty()) {
                        PendingReason::ReqNodeNotAvail
                    } else {
                        PendingReason::NodeDown
                    },
                );
                continue;
            }

            // Fewer nodes free (schedulable, available resources) than needed →
            // Resources; otherwise queued behind higher priority.
            let has_capacity = |n: &spur_core::node::Node| {
                n.has_free_cpu_capacity() && n.can_satisfy_request(&required)
            };
            let free_now = eligible
                .iter()
                .filter(|n| placement.matches(n, cluster_state.reservations, now))
                .filter(|n| has_capacity(n))
                .count();

            let reason = if free_now >= needed {
                PendingReason::Priority
            } else {
                // k0s-caused shortfall: nodes that would match but for the k0s gate.
                // Uses matches_ignoring_k0s so exclusive/idle rules stay in lockstep.
                let k0s_blocked = eligible
                    .iter()
                    .filter(|n| {
                        n.is_k0s_reserved()
                            && placement.matches_ignoring_k0s(n, cluster_state.reservations, now)
                            && has_capacity(n)
                    })
                    .count();
                if free_now + k0s_blocked >= needed {
                    PendingReason::K8sReserved
                } else {
                    PendingReason::Resources
                }
            };
            job_entry.set_pending_reason(reason);
        }
    }

    /// Send a job event notification via webhook (if configured).
    ///
    /// Uses `curl` as a subprocess to avoid pulling in an HTTP client dependency.
    fn send_notification(&self, job_id: JobId, event: &str, spec: &JobSpec) {
        let config = self.config();
        let webhook_url = config.notifications.webhook_url.clone();
        if let Some(url) = webhook_url {
            let event = event.to_string();
            let user = spec.user.clone();
            let mail_user = spec.mail_user.clone();
            let job_name = spec.name.clone();
            tokio::spawn(async move {
                let payload = serde_json::json!({
                    "job_id": job_id,
                    "event": event,
                    "job_name": job_name,
                    "user": user,
                    "mail_user": mail_user,
                });
                let payload_str = payload.to_string();
                match tokio::process::Command::new("curl")
                    .args([
                        "-s",
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        &payload_str,
                        &url,
                    ])
                    .output()
                    .await
                {
                    Ok(output) => {
                        if !output.status.success() {
                            tracing::warn!(
                                job_id,
                                %event,
                                "notification webhook returned non-zero exit"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            job_id,
                            %event,
                            error = %e,
                            "failed to send notification webhook"
                        );
                    }
                }
            });
        }

        // SMTP email notification via sendmail-compatible command
        if let Some(ref smtp_cmd) = config.notifications.smtp_command {
            let from = config
                .notifications
                .from_address
                .as_deref()
                .unwrap_or("spur@localhost");
            let user = spec.user.clone();
            let mail_user = spec.mail_user.clone();
            let to = mail_user.as_deref().unwrap_or(&user).to_string();
            let subject = format!("Spur Job {}: {}", job_id, event);
            let body = format!("Job ID: {}\nEvent: {}\nUser: {}\n", job_id, event, user);
            let email = format!(
                "From: {}\nTo: {}\nSubject: {}\n\n{}",
                from, to, subject, body
            );

            let smtp_cmd = smtp_cmd.clone();
            tokio::spawn(async move {
                let mut child = tokio::process::Command::new("sh")
                    .args(["-c", &smtp_cmd])
                    .stdin(std::process::Stdio::piped())
                    .spawn();
                if let Ok(ref mut child) = child {
                    if let Some(ref mut stdin) = child.stdin.take() {
                        use tokio::io::AsyncWriteExt;
                        let _ = stdin.write_all(email.as_bytes()).await;
                    }
                    let _ = child.wait().await;
                }
            });
        }
    }

    pub fn set_raft(&self, raft: SpurRaft) {
        *self.raft.write() = Some(raft);
    }

    pub fn set_accounting(&self, notifier: AccountingNotifier) {
        *self.accounting.write() = Some(notifier);
    }

    pub fn set_sched_stats(&self, stats: Arc<SchedStatsCollector>) {
        let _ = self.sched_stats.set(stats);
    }

    pub(crate) fn set_planned_reservations(
        &self,
        planned: HashMap<String, (JobId, DateTime<Utc>)>,
    ) {
        *self.planned_reservations.write() = planned;
    }

    /// `(job_id, start_time)` if `node` is idle right now but the scheduler
    /// holds a future reservation for a specific pending job on it.
    pub fn planned_reservation(&self, node: &str) -> Option<(JobId, DateTime<Utc>)> {
        self.planned_reservations.read().get(node).copied()
    }

    pub(crate) fn set_planned_job_starts(&self, planned: spur_sched::backfill::PlannedJobStarts) {
        *self.planned_job_starts.write() = planned;
    }

    /// Borrow the plan under the read lock; cloning it per job-info response
    /// would be avoidable per-request work as the pending queue grows.
    pub(crate) fn with_planned_job_starts<R>(
        &self,
        f: impl FnOnce(&spur_sched::backfill::PlannedJobStarts) -> R,
    ) -> R {
        f(&self.planned_job_starts.read())
    }

    pub(crate) fn record_sched_cycle(
        &self,
        cycle_time_us: u64,
        schedule_time_us: u64,
        jobs_started: u64,
        hit_depth_limit: bool,
    ) {
        if let Some(stats) = self.sched_stats.get() {
            stats.record_cycle(
                cycle_time_us,
                schedule_time_us,
                jobs_started,
                hit_depth_limit,
            );
        }
    }

    pub fn fairshare_cache(&self) -> &Arc<FairshareCache> {
        &self.fairshare_cache
    }

    pub fn qos_cache(&self) -> &Arc<QosCache> {
        &self.qos_cache
    }

    pub fn grp_wall_cache(&self) -> &Arc<GrpWallCache> {
        &self.grp_wall_cache
    }

    pub fn association_cache(&self) -> &Arc<AssociationCache> {
        &self.association_cache
    }

    /// Whether accounting is configured. A cluster with no `database_url` never
    /// loads caps, so a cold cache there means "no caps exist" rather than
    /// "caps could not be read".
    pub(crate) fn accounting_enabled(&self) -> bool {
        !self.config().accounting.database_url.is_empty()
    }

    /// Limits beside live usage, per user and scope, for `scontrol show assoc_mgr`.
    /// Usage always reflects the job table; caps appear only where a cache holds a
    /// snapshot, which `limits_readable` reports so an operator can tell an
    /// enabled-but-cold cache from an accounting-off cluster with no caps at all.
    /// `only_user` narrows the view to scopes that user takes part in, so a
    /// non-admin caller cannot enumerate the cluster-wide QOS/account inventory.
    pub(crate) fn assoc_mgr_info(&self, only_user: Option<&str>) -> AssocMgrInfo {
        let jobs = self.jobs.read();

        let defined_qos: HashMap<String, Qos> = self
            .qos_cache
            .all()
            .into_iter()
            .map(|q| (q.name.clone(), q))
            .collect();
        let mut qos_scopes: BTreeSet<String> = defined_qos.keys().cloned().collect();
        qos_scopes.extend(queued_scopes(&jobs, qos_of));

        let qos_records = qos_scopes
            .into_iter()
            .filter_map(|scope| {
                let limits = defined_qos.get(&scope).map(|q| q.limits.clone());
                // A QOS holds one set of per-user caps for everyone it governs.
                let user_caps = limits.as_ref().map(|l| PerUserCaps {
                    max_jobs: l.max_jobs_per_user,
                    max_submit_jobs: l.max_submit_jobs_per_user,
                    max_tres: l.max_tres_per_user.clone(),
                });
                let mut record = scope_usage(&jobs, &scope, qos_of, only_user, &[], |_| {
                    user_caps.clone().unwrap_or_default()
                })?;
                record.max_wall_minutes = limits.as_ref().and_then(|l| l.max_wall_minutes);
                record.grp_tres = limits.as_ref().and_then(|l| l.grp_tres.clone());
                record.grp_submit_jobs = limits.as_ref().and_then(|l| l.grp_submit_jobs);
                record.user_caps = user_caps;
                Some(record)
            })
            .collect();

        // An association's caps are per (user, account), so its users come from the
        // definitions as well as from the queue, and each carries its own caps.
        let mut defined_assoc: HashMap<String, Vec<(String, AccountLimits)>> = HashMap::new();
        for (account, user, limits) in self.association_cache.all() {
            defined_assoc
                .entry(account)
                .or_default()
                .push((user, limits));
        }
        let mut assoc_scopes: BTreeSet<String> = defined_assoc.keys().cloned().collect();
        assoc_scopes.extend(queued_scopes(&jobs, account_of));

        let assoc_records = assoc_scopes
            .into_iter()
            .filter_map(|scope| {
                let rows = defined_assoc.get(&scope);
                // The rows from `association_cache.all()` already carry each
                // (user, account) limit, so read the per-user and group caps from
                // them rather than re-locking the cache once per user.
                let by_user: HashMap<&str, &AccountLimits> = rows
                    .map(|rows| rows.iter().map(|(u, l)| (u.as_str(), l)).collect())
                    .unwrap_or_default();
                let defined_users: Vec<String> = by_user.keys().map(|u| (*u).to_owned()).collect();
                // An association carries no per-user TRES cap, so `max_tres` stays unset.
                let mut record = scope_usage(
                    &jobs,
                    &scope,
                    account_of,
                    only_user,
                    &defined_users,
                    |user| {
                        let limits = by_user.get(user);
                        PerUserCaps {
                            max_jobs: limits.and_then(|l| l.max_running_jobs),
                            max_submit_jobs: limits.and_then(|l| l.max_submit_jobs),
                            max_tres: None,
                        }
                    },
                )?;
                // The group caps are the account's, but the schema keys every
                // association row separately, so take them from the first row
                // rather than inventing a merge across rows that disagree.
                if let Some((_, limits)) = rows.and_then(|rows| rows.first()) {
                    record.max_wall_minutes = limits.max_wall_minutes;
                    record.grp_tres = limits.grp_tres.clone();
                    record.grp_submit_jobs = limits.grp_submit_jobs;
                }
                Some(record)
            })
            .collect();

        AssocMgrInfo {
            qos_records,
            assoc_records,
            limits_readable: limits_readable(
                self.accounting_enabled(),
                self.qos_cache.is_loaded(),
                self.association_cache.is_loaded(),
            ),
        }
    }

    /// Resolve a job's QoS from the cache; unknown/absent name → limitless default.
    /// Callers that gate admission must go through `accounting_block` first: the
    /// limitless default is only safe where a missing QOS costs nothing (priority
    /// and preemption comparisons), never where a limit is being applied.
    pub(crate) fn resolve_qos(&self, job: &Job) -> Qos {
        match job.spec.qos.as_deref() {
            Some(name) => self.qos_cache.get(name).unwrap_or_default(),
            None => Qos::default(),
        }
    }

    /// The wall-clock ceilings that already govern this submission, for defaulting.
    /// An unloaded cache reports nothing rather than holding the job, matching how
    /// `enforce_submit_limits` treats limits it cannot read yet.
    fn wall_caps(&self, spec: &JobSpec) -> WallCaps {
        let qos_minutes = spec
            .qos
            .as_deref()
            .filter(|n| !n.is_empty() && self.qos_cache.is_loaded())
            .and_then(|name| self.qos_cache.get(name))
            .and_then(|qos| qos.limits.max_wall_minutes);

        let account_minutes = spec
            .account
            .as_deref()
            .filter(|a| !a.is_empty() && self.association_cache.is_loaded())
            .and_then(|account| {
                self.association_cache
                    .limits(&spec.user, account)
                    .max_wall_minutes
            });

        WallCaps {
            qos_minutes,
            account_minutes,
        }
    }

    /// Whether QOS limits can be *read* right now: either accounting is off (no
    /// caps exist to read) or the QOS cache has loaded at least one snapshot.
    /// This is the canonical per-cache verdict `accounting_block` gates on; it
    /// is exposed so operator-facing surfaces report the same thing rather than
    /// re-deriving a weaker `loaded`-only check that misreports both the
    /// accounting-disabled case (no caps) and a split cache (one cache warm).
    pub(crate) fn qos_limits_readable(&self) -> bool {
        !self.config().accounting.enabled() || self.qos_cache.is_loaded()
    }

    /// Whether association limits can be read right now; the association-cache
    /// counterpart of [`qos_limits_readable`](Self::qos_limits_readable).
    pub(crate) fn association_limits_readable(&self) -> bool {
        !self.config().accounting.enabled() || self.association_cache.is_loaded()
    }

    /// Holds a candidate whose accounting limits cannot be read. Admission is the
    /// last point at which a cap can still be applied — a running job is never
    /// re-checked — so a cold cache pends the job rather than gating it against a
    /// limitless default.
    fn accounting_block(&self, job: &Job) -> Option<PendingReason> {
        if !self.config().accounting.enabled() {
            return None;
        }

        if let Some(name) = job.spec.qos.as_deref().filter(|n| !n.is_empty()) {
            if !self.qos_limits_readable() {
                return Some(PendingReason::AccountingUnavailable);
            }
            // Loaded and still absent means the QOS was removed after the job
            // queued, which Slurm reports the same way.
            if self.qos_cache.get(name).is_none() {
                return Some(PendingReason::InvalidQOS);
            }
        }

        let has_account = job.spec.account.as_deref().is_some_and(|a| !a.is_empty());
        if has_account && !self.association_limits_readable() {
            return Some(PendingReason::AccountingUnavailable);
        }

        None
    }

    /// Submit-time limit gate (Slurm's `acct_policy_validate`). Submit-count
    /// limits and the association per-job wall deny unconditionally; stand-alone
    /// TRES/wall breaches deny only under the governing QOS's `DenyOnLimit`
    /// (treated as on when the job has no QOS). `incoming` is the task count added.
    fn enforce_submit_limits(&self, spec: &JobSpec, incoming: u32) -> Result<(), SubmitError> {
        let jobs = self.jobs.read();
        let user = spec.user.as_str();
        let account = spec.account.as_deref().filter(|a| !a.is_empty());
        let qos_name = spec.qos.as_deref().filter(|n| !n.is_empty());

        let is_submitted = |j: &Job| matches!(j.state, JobState::Pending | JobState::Running);

        // Surface a human sentence plus the exact Slurm reason code, so the
        // rejection is readable while machine consumers keep the bare code.
        let deny = |reason: PendingReason| {
            SubmitError::invalid(format!("{} ({reason})", reason.submit_denial_message()))
        };

        // Association submit-count + unconditional wall.
        let assoc_loaded = self.association_cache.is_loaded();
        if assoc_loaded {
            if let Some(account) = account {
                let limits = self.association_cache.limits(user, account);
                // Only scan the job table for a count a set limit actually needs.
                let user_submitted = if limits.max_submit_jobs.is_some() {
                    jobs.values()
                        .filter(|j| {
                            is_submitted(j)
                                && j.spec.user == user
                                && j.spec.account.as_deref() == Some(account)
                        })
                        .count() as u32
                } else {
                    0
                };
                let account_submitted = if limits.grp_submit_jobs.is_some() {
                    jobs.values()
                        .filter(|j| is_submitted(j) && j.spec.account.as_deref() == Some(account))
                        .count() as u32
                } else {
                    0
                };
                if let Some(reason) = check_account_submit_limits(
                    &limits,
                    user_submitted,
                    account_submitted,
                    incoming,
                ) {
                    return Err(deny(reason));
                }
                if let Some(reason) = check_account_wall_limit(spec, &limits) {
                    return Err(deny(reason));
                }
            }
        }

        // QOS submit-count + (DenyOnLimit-gated) stand-alone resource breach.
        // A job with no QOS is treated as DenyOnLimit-on so an unsatisfiable
        // association request is rejected rather than pending forever.
        let mut deny_on_limit = true;
        if self.qos_cache.is_loaded() {
            if let Some(qos_name) = qos_name {
                if let Some(qos) = self.qos_cache.get(qos_name) {
                    deny_on_limit = qos.deny_on_limit;
                    // Only scan the job table for a count a set limit actually needs.
                    let user_submitted = if qos.limits.max_submit_jobs_per_user.is_some() {
                        jobs.values()
                            .filter(|j| {
                                is_submitted(j)
                                    && j.spec.user == user
                                    && j.spec.qos.as_deref() == Some(qos_name)
                            })
                            .count() as u32
                    } else {
                        0
                    };
                    let account_submitted = if qos.limits.max_submit_jobs_per_account.is_some() {
                        jobs.values()
                            .filter(|j| {
                                is_submitted(j)
                                    && j.spec.account.as_deref() == account
                                    && j.spec.qos.as_deref() == Some(qos_name)
                            })
                            .count() as u32
                    } else {
                        0
                    };
                    let qos_submitted = if qos.limits.grp_submit_jobs.is_some() {
                        jobs.values()
                            .filter(|j| is_submitted(j) && j.spec.qos.as_deref() == Some(qos_name))
                            .count() as u32
                    } else {
                        0
                    };
                    if let Some(reason) = check_qos_submit_limits(
                        &qos,
                        user_submitted,
                        account_submitted,
                        qos_submitted,
                        incoming,
                    ) {
                        return Err(deny(reason));
                    }
                    if deny_on_limit {
                        if let Some(reason) = check_qos_standalone_limits(spec, &qos) {
                            return Err(deny(reason));
                        }
                    }
                }
            }
        }

        // Association stand-alone TRES breach, gated the same way as QOS.
        if deny_on_limit && assoc_loaded {
            if let Some(account) = account {
                let limits = self.association_cache.limits(user, account);
                if let Some(reason) = check_account_standalone_limits(spec, &limits) {
                    return Err(deny(reason));
                }
            }
        }

        Ok(())
    }

    /// Recompute a job's live effective priority. A running job's stored
    /// `priority` is stale; this recalculates from age, fair-share, and tier.
    pub(crate) fn current_effective_priority(&self, job: &Job, partitions: &[Partition]) -> u32 {
        let now = Utc::now();
        let age_minutes = (now - job.submit_time).num_minutes().max(0);
        let partition_tier =
            spur_core::partition::max_priority_tier(job.spec.partition.as_deref(), partitions);
        let fair_share = self
            .fairshare_cache
            .get(&job.spec.user, job.spec.account.as_deref().unwrap_or(""));
        compute_effective_priority(job.priority, fair_share, age_minutes, partition_tier)
    }

    /// Persist a mutation via Raft consensus. The apply callback
    /// (`StateMachineApply`) handles in-memory state on all nodes.
    fn complete_job_steps(&self, job_id: &JobId, exit_code: i32, timestamp: DateTime<Utc>) {
        let mut steps = self.steps.write();
        for step in steps.values_mut() {
            if step.job_id == *job_id && !step.state.is_terminal() {
                step.state = if exit_code == 0 {
                    StepState::Completed
                } else {
                    StepState::Failed
                };
                step.exit_code = Some(exit_code);
                step.end_time = Some(timestamp);
            }
        }
        drop(steps);
        // Licenses are not returned here: usage is derived from running jobs, so a
        // job leaving the running set frees its licenses automatically.
    }

    /// Finalize steps for all evicted jobs returned by remove_node / health check.
    pub fn complete_evicted_steps(&self, evicted: &[JobFinalized]) {
        let now = Utc::now();
        for fin in evicted {
            self.complete_job_steps(&fin.job_id, fin.exit_code, now);
        }
    }

    #[allow(clippy::result_large_err)]
    fn propose(&self, op: WalOperation) -> anyhow::Result<ClientResponse> {
        let raft = self
            .raft
            .read()
            .clone()
            .expect("raft must be set before propose is called");
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async { raft.client_write(op).await })
        })
        .map(|res| res.data)
        .map_err(|e| anyhow::anyhow!("raft propose failed: {}", e))
    }

    /// Return a finished job's per-node slice to each node it held, skipping any
    /// in `already_deallocated`; pass `allocated_resources = None` if freed already.
    fn deallocate_job_slices(
        nodes: &mut HashMap<String, Node>,
        freed_nodes: &[String],
        allocated_resources: Option<&ResourceAllocations>,
        per_node_alloc: &HashMap<String, ResourceAllocations>,
        already_deallocated: &[String],
        job_id: JobId,
    ) {
        let Some(total) = allocated_resources else {
            return;
        };
        let node_count = freed_nodes.len().max(1) as u32;
        for name in freed_nodes {
            if already_deallocated.iter().any(|n| n == name) {
                continue;
            }
            let Some(node) = nodes.get_mut(name) else {
                continue;
            };
            let slice = per_node_alloc.get(name).cloned().unwrap_or_else(|| {
                warn!(job_id, node = %name, "per_node_alloc missing at deallocation, using scalar fallback");
                ResourceAllocations::with_scalar(
                    total.cpus / node_count,
                    total.memory_mb / node_count as u64,
                )
            });
            node.alloc_resources.subtract(&slice);
            node.update_state_from_alloc();
            if node.state == NodeState::Draining
                && node.alloc_resources.cpus == 0
                && !node.alloc_resources.has_devices()
            {
                node.state = NodeState::Drain;
            }
        }
    }

    /// Clear a job's run-state fields so it's schedulable again after requeue.
    /// Does not bump either counter; callers do that based on why they're requeuing.
    fn clear_run_state_for_requeue(job: &mut Job) {
        job.start_time = None;
        job.exit_code = None;
        job.allocated_nodes.clear();
        job.allocated_resources = None;
        job.per_node_alloc.clear();
        job.node_completions.clear();
        job.time_limit_signaled_at = None;
        job.pending_reason = PendingReason::None;
        job.pending_reason_desc = None;
        // Stale after requeue (points at nodes the job left); next dispatch resets it.
        job.actual_stdout_path = None;
        job.actual_stderr_path = None;
    }

    pub fn set_job_launch_failure_detail(
        &self,
        job_id: JobId,
        detail: String,
    ) -> anyhow::Result<()> {
        self.propose(WalOperation::JobLaunchFailureDetail { job_id, detail })?;
        Ok(())
    }

    /// Requeue after a dispatch failure or Timeout/NodeFail: counts against
    /// `max_batch_requeue`.
    fn reset_job_for_requeue(job: &mut Job) {
        job.requeue_count += 1;
        Self::clear_run_state_for_requeue(job);
    }

    /// Requeue after preemption: tracked separately since it isn't a failure
    /// signal and must never contribute to the `max_batch_requeue` hold.
    fn reset_job_for_preempt_requeue(job: &mut Job) {
        job.preempt_requeue_count += 1;
        Self::clear_run_state_for_requeue(job);
    }

    /// Requeue by admin action (`scontrol requeue`): tracked separately since it
    /// is an operator decision, never a failure, and must never contribute to
    /// the `max_batch_requeue` hold.
    fn reset_job_for_user_requeue(job: &mut Job) {
        job.user_requeue_count += 1;
        Self::clear_run_state_for_requeue(job);
    }

    /// Evict a single job by ID: transition to NodeFail, then free its
    /// allocations on every node it spans. Transition is validated first
    /// so allocations are never freed for a job that can't be evicted.
    /// Nodes that already reported completion via `JobNodeComplete` (which
    /// frees a node's slice as it arrives, ahead of the whole job finishing)
    /// are skipped so their resources aren't subtracted twice.
    fn evict_job_locked(
        job_id: JobId,
        jobs: &mut HashMap<JobId, Job>,
        nodes: &mut HashMap<String, Node>,
        timestamp: chrono::DateTime<Utc>,
        reason: PendingReason,
    ) -> Option<JobFinalized> {
        let job = jobs.get_mut(&job_id)?;

        if let Some(since) = job.suspended_at.take() {
            job.suspended_secs += (timestamp - since).num_seconds().max(0);
        }
        if let Err(e) = job.transition(JobState::NodeFail) {
            warn!(job_id, error = %e, "evict: invalid transition to NodeFail");
            return None;
        }
        job.exit_code = Some(-1);
        job.end_time = Some(timestamp);
        job.set_pending_reason(reason);
        let already_deallocated: Vec<String> = job.node_completions.keys().cloned().collect();
        job.node_completions.clear();

        Self::deallocate_job_slices(
            nodes,
            &job.allocated_nodes,
            job.allocated_resources.as_ref(),
            &job.per_node_alloc,
            &already_deallocated,
            job_id,
        );

        Some(JobFinalized {
            job_id,
            state: JobState::NodeFail,
            exit_code: -1,
        })
    }

    /// Fail all running/completing/suspended jobs on a node, releasing
    /// allocations on **every** node each job spans.
    fn evict_jobs_on_node(
        node_name: &str,
        jobs: &mut HashMap<JobId, Job>,
        nodes: &mut HashMap<String, Node>,
        timestamp: chrono::DateTime<Utc>,
        response: &mut ClientResponse,
    ) {
        let affected: Vec<JobId> = jobs
            .iter()
            .filter(|(_, j)| {
                matches!(
                    j.state,
                    JobState::Running | JobState::Completing | JobState::Suspended
                ) && j.allocated_nodes.iter().any(|n| n == node_name)
            })
            .map(|(&id, _)| id)
            .collect();

        for jid in affected {
            if let Some(fin) =
                Self::evict_job_locked(jid, jobs, nodes, timestamp, PendingReason::NodeDown)
            {
                response.jobs_finalized.push(fin);
            }
        }
    }

    /// Apply a WalOperation to in-memory state.
    /// Called by Raft's `apply_to_state_machine` on commit.
    fn apply_operation(&self, op: &WalOperation) -> ClientResponse {
        let mut response = ClientResponse::default();
        let mut jobs = self.jobs.write();
        let mut nodes = self.nodes.write();
        let mut next_id = self.next_job_id.load(Ordering::Relaxed);
        let timestamp = Utc::now();

        match op {
            WalOperation::JobSubmit { job_id, spec } => {
                let mut job = Job::new(*job_id, (**spec).clone());
                if let Some(het_group) = spec.het_group {
                    job.het_group = Some(het_group);
                    if het_group > 0 {
                        let anchor = jobs.values().find(|j| {
                            j.het_group == Some(0)
                                && j.spec.user == spec.user
                                && j.spec.name == spec.name
                                && j.state == JobState::Pending
                        });
                        if let Some(a) = anchor {
                            job.het_job_id = Some(a.job_id);
                        }
                    }
                }
                jobs.insert(*job_id, job);
                next_id = next_id.max(job_id + 1);
            }
            WalOperation::JobStateChange {
                job_id,
                new_state,
                pending_reason,
                pending_priority,
                begin_time,
                pending_reason_desc,
                ..
            } => {
                if let Some(job) = jobs.get_mut(job_id) {
                    let outcome = match job.apply_transition(*new_state) {
                        Ok(outcome) => outcome,
                        Err(e) => {
                            warn!(job_id = *job_id, error = %e, "invalid state transition in WAL apply");
                            TransitionOutcome::NoOp
                        }
                    };
                    // Gated on a real transition so a replay doesn't re-wipe
                    // fields or double-count requeue_count.
                    if outcome == TransitionOutcome::Applied && *new_state == JobState::Pending {
                        let max = self.config().controller.max_batch_requeue;
                        if job.requeue_count < max {
                            Self::reset_job_for_requeue(job);
                        } else {
                            Self::clear_run_state_for_requeue(job);
                        }
                        let reason = pending_reason.clone().unwrap_or(PendingReason::None);
                        match pending_reason_desc {
                            Some(desc) => job.set_pending_reason_desc(reason, desc.clone()),
                            None => job.set_pending_reason(reason),
                        }
                        if let Some(priority) = pending_priority {
                            job.priority = *priority;
                        }
                        if let Some(hold) = begin_time {
                            job.spec.begin_time = Some(*hold);
                        }
                    }
                }
            }
            WalOperation::JobDispatchBackoff { job_id, begin_time } => {
                // NoOp if the job left Pending since the leader proposed this
                // (e.g. a concurrent cancel).
                let Some(job) = jobs.get_mut(job_id) else {
                    return ClientResponse::default();
                };
                if job.state != JobState::Pending {
                    return ClientResponse::default();
                }
                Self::reset_job_for_requeue(job);
                job.spec.begin_time = Some(*begin_time);
                job.set_pending_reason(PendingReason::JobLaunchFailure);
            }
            WalOperation::JobPreemptRequeue {
                job_id,
                begin_time,
                preempted_by,
                preempt_qos,
            } => {
                // Only a running job is preempted; on replay the job is already
                // Pending, so this is a NoOp (no re-dealloc, no double requeue).
                let freed_nodes;
                let allocated_resources;
                let per_node_map;
                {
                    let Some(job) = jobs.get_mut(job_id) else {
                        return ClientResponse::default();
                    };
                    if job.state != JobState::Running {
                        return ClientResponse::default();
                    }
                    // Route through Preempted so the state machine and accounting
                    // see a finished run, then requeue to Pending — one atomic
                    // apply; the intermediate Preempted never escapes the lock.
                    if let Err(e) = job.transition(JobState::Preempted) {
                        warn!(job_id = *job_id, error = %e, "invalid preempt transition in WAL apply");
                        return ClientResponse::default();
                    }
                    job.exit_code = Some(-1);
                    job.end_time = Some(timestamp);
                    if let Some(since) = job.suspended_at.take() {
                        job.suspended_secs += (timestamp - since).num_seconds().max(0);
                    }
                    freed_nodes = job.allocated_nodes.clone();
                    allocated_resources = job.allocated_resources.clone();
                    per_node_map = job.per_node_alloc.clone();
                    job.node_completions.clear();

                    if let Err(e) = job.transition(JobState::Pending) {
                        warn!(job_id = *job_id, error = %e, "invalid requeue transition in WAL apply");
                        return ClientResponse::default();
                    }
                    Self::reset_job_for_preempt_requeue(job);
                    job.spec.begin_time = Some(*begin_time);
                    // Provenance fields are set after reset_job_for_preempt_requeue
                    // clears run state so they survive into the pending phase.
                    job.preempted_by = *preempted_by;
                    job.preempt_mode = Some("Requeue".to_string());
                    job.preempt_qos = preempt_qos.clone();
                    let desc = match preempted_by {
                        Some(by) => match preempt_qos.as_deref() {
                            Some(qos) => format!("preempted by job {by} (QOS: {qos})"),
                            None => format!("preempted by job {by}"),
                        },
                        None => "preempted".to_string(),
                    };
                    job.set_pending_reason_desc(PendingReason::Preempted, desc);
                }
                Self::deallocate_job_slices(
                    &mut nodes,
                    &freed_nodes,
                    allocated_resources.as_ref(),
                    &per_node_map,
                    &[],
                    *job_id,
                );
                drop(jobs);
                drop(nodes);
                // Fire accounting for the terminated run as PREEMPTED, even
                // though the job itself is now Pending-with-hold.
                self.complete_job_steps(job_id, -1, timestamp);
                self.next_job_id.store(next_id, Ordering::Relaxed);
                return ClientResponse {
                    jobs_finalized: vec![JobFinalized {
                        job_id: *job_id,
                        state: JobState::Preempted,
                        exit_code: -1,
                    }],
                    ..Default::default()
                };
            }
            WalOperation::JobUserRequeue {
                job_id,
                hold,
                begin_time,
            } => {
                // A live job is finalized once (routed through Requeued so
                // accounting/steps see a finished run) then re-pended; an
                // already-terminal job re-pends with no second finalization to
                // avoid double-counting. NoOp on replay / non-requeuable states.
                let was_live;
                let freed_nodes;
                let allocated_resources;
                let per_node_map;
                {
                    let Some(job) = jobs.get_mut(job_id) else {
                        return ClientResponse::default();
                    };
                    match job.state {
                        JobState::Running | JobState::Suspended => {
                            was_live = true;
                            if let Some(since) = job.suspended_at.take() {
                                job.suspended_secs += (timestamp - since).num_seconds().max(0);
                            }
                            // The live run still holds its allocation; capture it
                            // to free below (reset clears node_completions).
                            freed_nodes = job.allocated_nodes.clone();
                            allocated_resources = job.allocated_resources.clone();
                            per_node_map = job.per_node_alloc.clone();
                        }
                        s if s.is_terminal() => {
                            // The slice was already freed at completion; freeing
                            // again would double-subtract on any shared node.
                            was_live = false;
                            freed_nodes = Vec::new();
                            allocated_resources = None;
                            per_node_map = HashMap::new();
                        }
                        _ => {
                            // Pending, Completing, or an in-flight finalized
                            // state: nothing to requeue.
                            return ClientResponse::default();
                        }
                    }

                    // Guarded re-pend: a live run finalizes as Requeued first,
                    // then returns to Pending (terminal jobs go straight back).
                    if let Err(e) = job.requeue_to_pending() {
                        warn!(job_id = *job_id, error = %e, "invalid requeue transition in WAL apply");
                        return ClientResponse::default();
                    }
                    Self::reset_job_for_user_requeue(job);
                    if *hold {
                        job.priority = 0;
                        job.set_pending_reason(PendingReason::Held);
                    } else if let Some(bt) = begin_time {
                        // Leader-computed hold against re-dispatch into the cancel.
                        job.spec.begin_time = Some(*bt);
                        job.set_pending_reason(PendingReason::BeginTime);
                    } else {
                        job.set_pending_reason(PendingReason::None);
                    }
                }
                // Live path only; the terminal path passes an empty allocation so
                // this is a no-op (its slice was already freed at completion).
                Self::deallocate_job_slices(
                    &mut nodes,
                    &freed_nodes,
                    allocated_resources.as_ref(),
                    &per_node_map,
                    &[],
                    *job_id,
                );
                drop(jobs);
                drop(nodes);
                self.next_job_id.store(next_id, Ordering::Relaxed);
                if was_live {
                    // Complete steps and fire accounting-end for the terminated
                    // run as REQUEUED; the terminal path already did this at its
                    // real completion, so it emits nothing here (no double-count).
                    self.complete_job_steps(job_id, -1, timestamp);
                    return ClientResponse {
                        jobs_finalized: vec![JobFinalized {
                            job_id: *job_id,
                            state: JobState::Requeued,
                            exit_code: -1,
                        }],
                        ..Default::default()
                    };
                }
                return ClientResponse::default();
            }
            WalOperation::JobPreemptCancel {
                job_id,
                preempted_by,
                preempt_qos,
            } => {
                let freed_nodes;
                let allocated_resources;
                let per_node_map;
                {
                    let Some(job) = jobs.get_mut(job_id) else {
                        return ClientResponse::default();
                    };
                    if job.state != JobState::Running {
                        return ClientResponse::default();
                    }
                    if let Err(e) = job.transition(JobState::Preempted) {
                        warn!(job_id = *job_id, error = %e, "invalid preempt-cancel transition in WAL apply");
                        return ClientResponse::default();
                    }
                    job.exit_code = Some(-1);
                    job.end_time = Some(timestamp);
                    if let Some(since) = job.suspended_at.take() {
                        job.suspended_secs += (timestamp - since).num_seconds().max(0);
                    }
                    freed_nodes = job.allocated_nodes.clone();
                    allocated_resources = job.allocated_resources.clone();
                    per_node_map = job.per_node_alloc.clone();
                    job.node_completions.clear();
                    // Cancelled is terminal (dependents/arrays unblock); Preempted
                    // is reported to accounting via jobs_finalized.
                    if let Err(e) = job.transition(JobState::Cancelled) {
                        warn!(job_id = *job_id, error = %e, "invalid preempt-cancel final transition in WAL apply");
                        return ClientResponse::default();
                    }
                    job.preempted_by = *preempted_by;
                    job.preempt_mode = Some("Cancel".to_string());
                    job.preempt_qos = preempt_qos.clone();
                }
                if let Some(ref total) = allocated_resources {
                    let node_count = freed_nodes.len().max(1) as u32;
                    for name in &freed_nodes {
                        if let Some(node) = nodes.get_mut(name) {
                            let slice = per_node_map.get(name).cloned().unwrap_or_else(|| {
                                warn!(
                                    job_id = *job_id,
                                    node = %name,
                                    "per_node_alloc missing at preempt-cancel deallocation, using scalar fallback"
                                );
                                ResourceAllocations::with_scalar(
                                    total.cpus / node_count,
                                    total.memory_mb / node_count as u64,
                                )
                            });
                            node.alloc_resources.subtract(&slice);
                            node.update_state_from_alloc();
                            if node.state == NodeState::Draining
                                && node.alloc_resources.cpus == 0
                                && !node.alloc_resources.has_devices()
                            {
                                node.state = NodeState::Drain;
                            }
                        }
                    }
                }
                drop(jobs);
                drop(nodes);
                self.complete_job_steps(job_id, -1, timestamp);
                self.next_job_id.store(next_id, Ordering::Relaxed);
                return ClientResponse {
                    jobs_finalized: vec![JobFinalized {
                        job_id: *job_id,
                        state: JobState::Preempted,
                        exit_code: -1,
                    }],
                    ..Default::default()
                };
            }
            WalOperation::JobSuspend {
                job_id,
                at,
                preempted_by,
                preempt_qos,
            } => {
                if let Some(job) = jobs.get_mut(job_id) {
                    match job.apply_transition(JobState::Suspended) {
                        Ok(TransitionOutcome::Applied) => {
                            job.suspended_at = Some(*at);
                            if preempted_by.is_some() {
                                job.preempted_by = *preempted_by;
                                job.preempt_mode = Some("Suspend".to_string());
                                job.preempt_qos = preempt_qos.clone();
                            }
                        }
                        Ok(TransitionOutcome::NoOp) => {}
                        Err(e) => {
                            warn!(job_id = *job_id, error = %e, "invalid suspend transition in WAL apply")
                        }
                    }
                }
            }
            WalOperation::JobResume { job_id, at } => {
                if let Some(job) = jobs.get_mut(job_id) {
                    match job.apply_transition(JobState::Running) {
                        Ok(TransitionOutcome::Applied) => {
                            if let Some(since) = job.suspended_at.take() {
                                job.suspended_secs += (*at - since).num_seconds().max(0);
                            }
                            // Resuming clears preemption provenance: the job is
                            // running again and a subsequent normal completion
                            // must not report it as preempted in accounting.
                            job.preempted_by = None;
                            job.preempt_mode = None;
                            job.preempt_qos = None;
                        }
                        Ok(TransitionOutcome::NoOp) => {}
                        Err(e) => {
                            warn!(job_id = *job_id, error = %e, "invalid resume transition in WAL apply")
                        }
                    }
                }
            }
            WalOperation::JobEvict { job_id, detail } => {
                if let Some(job) = jobs.get_mut(job_id) {
                    job.launch_failure_detail = detail.clone();
                }
                if let Some(fin) = Self::evict_job_locked(
                    *job_id,
                    &mut jobs,
                    &mut nodes,
                    timestamp,
                    PendingReason::JobLaunchFailure,
                ) {
                    response.jobs_finalized.push(fin);
                }
            }
            WalOperation::JobLaunchFailureDetail { job_id, detail } => {
                if let Some(job) = jobs.get_mut(job_id) {
                    job.launch_failure_detail = Some(detail.clone());
                }
            }
            WalOperation::JobStart {
                job_id,
                nodes: node_names,
                resources,
                per_node_alloc,
                srun_step_dispatch,
                run_attempt,
            } => {
                if let Some(job) = jobs.get_mut(job_id) {
                    job.start_time = Some(timestamp);
                    job.allocated_nodes = node_names.clone();
                    job.allocated_resources = Some(resources.clone());
                    job.per_node_alloc = per_node_alloc.clone();
                    job.set_pending_reason(PendingReason::None);
                    job.srun_step_dispatch = *srun_step_dispatch;
                    job.run_attempt = *run_attempt;
                    job.launch_failure_detail = None;
                    // A new run supersedes any prior preemption provenance; clear so
                    // this run's accounting record does not inherit the previous one's.
                    job.preempted_by = None;
                    job.preempt_mode = None;
                    job.preempt_qos = None;
                    // Same reason, and it must be here rather than at requeue:
                    // the finished run's record is written after that apply.
                    job.derived_exit_code = 0;
                    job.exit_signal = 0;
                }
                let node_count = node_names.len().max(1) as u32;
                for name in node_names {
                    if let Some(node) = nodes.get_mut(name) {
                        let slice = per_node_alloc.get(name).cloned().unwrap_or_else(|| {
                            warn!(job_id = *job_id, node = %name, "per_node_alloc missing at allocation, using scalar fallback");
                            ResourceAllocations::with_scalar(
                                resources.cpus / node_count,
                                resources.memory_mb / node_count as u64,
                            )
                        });
                        node.alloc_resources.add(&slice);
                        node.update_state_from_alloc();
                    }
                }
                // Licenses are not mutated here: usage is derived on demand from
                // running jobs (see available_licenses()), so the config total is
                // authoritative and cannot drift.
            }
            WalOperation::JobNodeComplete {
                job_id,
                node_name,
                exit_code,
                signal,
            } => {
                let finalized = {
                    let Some(job) = jobs.get_mut(job_id) else {
                        return ClientResponse::default();
                    };
                    // A completion for a non-active job is stale/replayed; skip
                    // it rather than forcing an illegal finalize transition.
                    if !job.state.is_active() {
                        return ClientResponse::default();
                    }

                    let already_reported = job.node_completions.contains_key(node_name);
                    job.node_completions.insert(
                        node_name.clone(),
                        spur_core::job::NodeCompletion {
                            code: *exit_code,
                            signal: *signal,
                        },
                    );

                    if let Some(ref total) = job.allocated_resources {
                        if !already_reported {
                            let node_count = job.allocated_nodes.len().max(1) as u32;
                            if let Some(node) = nodes.get_mut(node_name) {
                                let slice = job.per_node_alloc.get(node_name).cloned().unwrap_or_else(|| {
                                    warn!(job_id = *job_id, node = %node_name, "per_node_alloc missing at node deallocation, using scalar fallback");
                                    ResourceAllocations::with_scalar(
                                        total.cpus / node_count,
                                        total.memory_mb / node_count as u64,
                                    )
                                });
                                node.alloc_resources.subtract(&slice);
                                node.update_state_from_alloc();
                                if node.state == NodeState::Draining
                                    && node.alloc_resources.cpus == 0
                                    && !node.alloc_resources.has_devices()
                                {
                                    node.state = NodeState::Drain;
                                }
                            }
                        }
                    }

                    // Suspended jobs route through Completing too, so an
                    // out-of-band task death finalizes instead of stranding.
                    if matches!(job.state, JobState::Running | JobState::Suspended) {
                        if let Err(e) = job.transition(JobState::Completing) {
                            warn!(job_id = *job_id, error = %e, "invalid transition to Completing");
                        }
                        job.end_time = Some(timestamp);
                    }

                    if job.all_nodes_completed() {
                        // Primary = batch node (allocated_nodes[0]); empty when
                        // none allocated, where derived_completion falls back to
                        // the worst completion.
                        let primary = job.allocated_nodes.first().cloned().unwrap_or_default();
                        // spurd flags an OOM kill via a sentinel bit in the signal;
                        // detect it, then strip the bit so the stored signal is the
                        // real SIGKILL and the job reports OUT_OF_MEMORY.
                        let oom = job
                            .node_completions
                            .values()
                            .any(|c| c.signal & spur_core::job::OOM_SIGNAL_FLAG != 0);
                        let (derived_state, final_exit, raw_signal) =
                            Job::derived_completion(&job.node_completions, &primary);
                        let final_signal = raw_signal & !spur_core::job::OOM_SIGNAL_FLAG;
                        let (final_state, final_reason) =
                            job.completion_verdict(derived_state, final_exit, final_signal, oom);
                        match job.transition(final_state) {
                            Ok(()) => {
                                job.exit_code = Some(final_exit);
                                job.exit_signal = final_signal;
                                // DerivedExitCode is the running max over srun
                                // steps, accumulated live by JobStepComplete; a
                                // job with no srun steps keeps 0 (Slurm parity),
                                // not the batch exit. Left as-is here.
                                job.set_pending_reason(final_reason);
                                job.end_time = Some(timestamp);
                                job.node_completions.clear();
                                Some((final_state, final_exit))
                            }
                            Err(e) => {
                                warn!(
                                    job_id = *job_id,
                                    error = %e,
                                    "invalid final completion transition"
                                );
                                None
                            }
                        }
                    } else {
                        None
                    }
                };

                if let Some((final_state, final_exit)) = finalized {
                    drop(jobs);
                    drop(nodes);
                    self.complete_job_steps(job_id, final_exit, timestamp);
                    self.next_job_id.store(next_id, Ordering::Relaxed);
                    return ClientResponse {
                        jobs_finalized: vec![JobFinalized {
                            job_id: *job_id,
                            state: final_state,
                            exit_code: final_exit,
                        }],
                        ..Default::default()
                    };
                }
            }
            WalOperation::JobTimeLimitSignaled { job_id, at } => {
                if let Some(job) = jobs.get_mut(job_id) {
                    // A run that already ended keeps the verdict it finalized
                    // with: the watchdog raced the job's own exit and lost.
                    if job.state.is_active() && job.time_limit_signaled_at.is_none() {
                        job.time_limit_signaled_at = Some(*at);
                    }
                }
            }
            WalOperation::JobComplete {
                job_id,
                exit_code,
                state,
            } => {
                let freed_nodes;
                let allocated_resources;
                let already_deallocated;
                if let Some(job) = jobs.get_mut(job_id) {
                    // is_finalized (incl. Preempted): a stale/replayed complete
                    // is a silent no-op, not a rejected-transition warning.
                    // Preempted is finalized for the ended run but may still cancel.
                    if job.state.is_finalized()
                        && !(job.state == JobState::Preempted && *state == JobState::Cancelled)
                    {
                        return ClientResponse::default();
                    }
                    if let Err(e) = job.transition(*state) {
                        warn!(
                            job_id = *job_id,
                            error = %e,
                            "invalid state transition in WAL apply"
                        );
                        return ClientResponse::default();
                    }
                    if state.is_terminal() {
                        response.jobs_finalized.push(JobFinalized {
                            job_id: *job_id,
                            state: *state,
                            exit_code: *exit_code,
                        });
                    }
                    job.exit_code = Some(*exit_code);
                    job.end_time = Some(timestamp);
                    // Derived from the replicated entry, so every replica reports
                    // the same reason for a job the watchdog had to force-kill.
                    if *state == JobState::Timeout {
                        job.set_pending_reason(PendingReason::TimeLimit);
                    }
                    // Suspended -> terminal: fold the final suspended interval in
                    // and clear suspended_at so it never lingers on a terminal job.
                    if let Some(since) = job.suspended_at.take() {
                        job.suspended_secs += (timestamp - since).num_seconds().max(0);
                    }
                    freed_nodes = job.allocated_nodes.clone();
                    allocated_resources = job.allocated_resources.clone();
                    already_deallocated = job.node_completions.keys().cloned().collect::<Vec<_>>();
                    job.node_completions.clear();
                } else {
                    return ClientResponse::default();
                }
                // Deallocate node resources not already freed during COMPLETING.
                let per_node_map = jobs
                    .get(job_id)
                    .map(|j| j.per_node_alloc.clone())
                    .unwrap_or_default();
                Self::deallocate_job_slices(
                    &mut nodes,
                    &freed_nodes,
                    allocated_resources.as_ref(),
                    &per_node_map,
                    &already_deallocated,
                    *job_id,
                );
                drop(jobs);
                drop(nodes);
                self.complete_job_steps(job_id, *exit_code, timestamp);
            }
            WalOperation::JobStepComplete {
                job_id,
                step_id,
                exit_code,
            } => {
                // Keyed on job state, not step state: the eviction sweep
                // (complete_evicted_steps) runs leader-only, so steps can diverge.
                let job_over = jobs.get(job_id).is_some_and(|j| j.state.is_terminal());
                if !job_over {
                    // Record the step's own exit code/state.
                    {
                        let mut steps = self.steps.write();
                        if let Some(step) = steps.get_mut(&(*job_id, *step_id)) {
                            step.state = if *exit_code == 0 {
                                StepState::Completed
                            } else {
                                StepState::Failed
                            };
                            step.exit_code = Some(*exit_code);
                            step.end_time = Some(timestamp);
                        }
                    }
                    // DerivedExitCode is the running max over srun steps (the batch
                    // step is excluded — it carries the job's own exit, not a step
                    // result). Maintained live so `scontrol show job` reflects it
                    // mid-run, matching Slurm.
                    if *step_id < STEP_RESERVED_MIN {
                        if let Some(job) = jobs.get_mut(job_id) {
                            job.derived_exit_code = job.derived_exit_code.max(*exit_code);
                        }
                    }
                }
            }
            WalOperation::JobStepCreate { step } => match jobs.get(&step.job_id) {
                None => {
                    warn!(
                        job_id = step.job_id,
                        step_id = step.step_id,
                        "JobStepCreate: skipping step for unknown job"
                    );
                }
                Some(job) if job.state.is_terminal() => {
                    warn!(
                        job_id = step.job_id,
                        step_id = step.step_id,
                        state = ?job.state,
                        "JobStepCreate: skipping step for terminal job"
                    );
                }
                Some(_) => {
                    let mut steps = self.steps.write();
                    steps
                        .entry((step.job_id, step.step_id))
                        .or_insert_with(|| (**step).clone());
                }
            },
            WalOperation::JobPriorityChange {
                job_id,
                new_priority,
                pending_reason,
                pending_reason_desc,
                reset_requeue_count,
                clear_reservation,
                ..
            } => {
                if let Some(job) = jobs.get_mut(job_id) {
                    job.priority = *new_priority;
                    if let Some(reason) = pending_reason {
                        match pending_reason_desc {
                            Some(desc) => job.set_pending_reason_desc(reason.clone(), desc.clone()),
                            None => job.set_pending_reason(reason.clone()),
                        }
                    }
                    if *reset_requeue_count {
                        job.requeue_count = 0;
                    }
                    if *clear_reservation {
                        job.spec.reservation = None;
                    }
                }
            }
            WalOperation::NodeRegister {
                name,
                hostname,
                resources,
                address,
                port,
                wg_pubkey,
                version,
                labels,
                source,
            } => {
                let mut node = Node::new(name.clone(), resources.clone());
                node.hostname = if hostname.is_empty() {
                    name.clone()
                } else {
                    hostname.clone()
                };
                node.labels = labels.clone();
                self.apply_node_config_policy(&mut node);
                if !address.is_empty() {
                    node.address = Some(address.clone());
                }
                node.port = *port;
                if !wg_pubkey.is_empty() {
                    node.wg_pubkey = Some(wg_pubkey.clone());
                }
                if !version.is_empty() {
                    node.version = Some(version.clone());
                }
                node.source = spur_core::node::resolve_wal_node_source(source, version, labels);
                node.last_heartbeat = Some(Utc::now());
                node.state = node
                    .state
                    .transition(&NodeEvent::Register, false)
                    .unwrap_or(NodeState::Idle);

                // Assign partitions: match by hostlist OR label selector (union)
                drop(nodes);
                let partitions = self.partitions.read();
                for part in partitions.iter() {
                    if partition_matches_node(part, name, labels) {
                        node.partitions.push(part.name.clone());
                    }
                }
                if node.partitions.is_empty() {
                    if let Some(dp) = partitions.iter().find(|p| p.is_default) {
                        node.partitions.push(dp.name.clone());
                    } else if let Some(first) = partitions.first() {
                        node.partitions.push(first.name.clone());
                    }
                }
                drop(partitions);

                let mut nodes = self.nodes.write();
                nodes.insert(name.clone(), node);
                self.next_job_id.store(next_id, Ordering::Relaxed);
                return ClientResponse::default();
            }
            WalOperation::NodeUpdate {
                name,
                hostname,
                resources,
                address,
                port,
                wg_pubkey,
                version,
                source,
            } => {
                if let Some(node) = nodes.get_mut(name) {
                    node.total_resources = resources.clone();
                    if !hostname.is_empty() {
                        node.hostname = hostname.clone();
                    }
                    if !address.is_empty() {
                        node.address = Some(address.clone());
                    }
                    node.port = *port;
                    if !wg_pubkey.is_empty() {
                        node.wg_pubkey = Some(wg_pubkey.clone());
                    }
                    if !version.is_empty() {
                        node.version = Some(version.clone());
                    }
                    node.source =
                        spur_core::node::resolve_wal_node_source(source, version, &node.labels);
                    node.last_heartbeat = Some(Utc::now());
                }
            }
            WalOperation::NodeStateChange {
                name,
                new_state,
                reason,
                admin_locked,
                reason_uid,
                reason_time,
                ..
            } => {
                if let Some(node) = nodes.get_mut(name) {
                    node.state = *new_state;
                    node.state_reason = reason.clone();
                    node.reason_uid = *reason_uid;
                    node.reason_time = *reason_time;
                    node.admin_locked = *admin_locked;
                }
                if *new_state == NodeState::Down {
                    Self::evict_jobs_on_node(name, &mut jobs, &mut nodes, timestamp, &mut response);
                }
            }
            WalOperation::NodeLabelsUpdate { name, set, remove } => {
                if let Some(node) = nodes.get_mut(name) {
                    for (k, v) in set {
                        node.labels.insert(k.clone(), v.clone());
                    }
                    for k in remove {
                        node.labels.remove(k);
                    }
                    // Re-evaluate partition membership after label change
                    let partitions = self.partitions.read();
                    let mut matched = Vec::new();
                    for part in partitions.iter() {
                        if partition_matches_node(part, &node.name, &node.labels) {
                            matched.push(part.name.clone());
                        }
                    }
                    if matched.is_empty() {
                        if let Some(dp) = partitions.iter().find(|p| p.is_default) {
                            matched.push(dp.name.clone());
                        } else if let Some(first) = partitions.first() {
                            matched.push(first.name.clone());
                        }
                    }
                    node.partitions = matched;

                    self.apply_node_config_policy(node);
                }
            }
            WalOperation::NodeRemove { name, reason } => {
                Self::evict_jobs_on_node(name, &mut jobs, &mut nodes, timestamp, &mut response);
                if let Some(node) = nodes.get(name) {
                    if node.alloc_resources.cpus > 0 || node.alloc_resources.has_devices() {
                        warn!(
                            node = %name,
                            reason = reason.as_deref().unwrap_or(""),
                            "removing node with nonzero allocations"
                        );
                    }
                    // A node can be force-removed while still k0s-assigned (no prior `k8s down`).
                    if let Some(role) = node.k0s_role {
                        self.k0s_role_counts.dec(role);
                    }
                }
                nodes.remove(name);
                info!(
                    node = %name,
                    reason = reason.as_deref().unwrap_or(""),
                    "node removed from cluster"
                );
            }
            WalOperation::TokenCreate { token } => {
                self.tokens.write().insert(token.id.clone(), token.clone());
            }
            WalOperation::TokenRevoke { token_id } => {
                if let Some(t) = self.tokens.write().get_mut(token_id) {
                    t.revoked = true;
                }
            }
            WalOperation::PartitionCreate { partition } => {
                // A create always clears any tombstone for this name — the admin
                // is explicitly recreating a previously-deleted partition.
                self.deleted_partition_names.write().remove(&partition.name);
                {
                    let mut partitions = self.partitions.write();
                    let existing = partitions.iter().position(|p| p.name == partition.name);
                    // Seeded name => the existing entry is only the pre-replay
                    // config seed, so the WAL entry overrides it; else duplicate.
                    let seeded = self
                        .config_seeded_partitions
                        .write()
                        .remove(&partition.name);
                    match existing {
                        Some(idx) if seeded => {
                            if partition.is_default {
                                for p in partitions.iter_mut() {
                                    p.is_default = false;
                                }
                            }
                            partitions[idx] = partition.clone();
                            response.partition_created = true;
                            info!(name = %partition.name, "partition restored from WAL over config seed");
                        }
                        Some(_) => {
                            warn!(
                                name = %partition.name,
                                "duplicate partition create in WAL apply, ignoring"
                            );
                        }
                        None => {
                            // Promote exactly one default: clear all others when
                            // the new partition is created as the default.
                            if partition.is_default {
                                for p in partitions.iter_mut() {
                                    p.is_default = false;
                                }
                            }
                            partitions.push(partition.clone());
                            response.partition_created = true;
                            info!(name = %partition.name, "partition created");
                        }
                    }
                }
                // Node-to-partition membership is derived from the partition
                // table (nodes/selector match), so a create can immediately
                // change which partitions already-registered nodes belong to.
                self.reconcile_partitions(&mut nodes);
            }
            WalOperation::PartitionUpdate {
                name,
                nodes: new_hostlist,
                selector,
                state,
                max_time_minutes,
                default_time_minutes,
                max_nodes,
                min_nodes,
                allow_accounts,
                allow_groups,
                deny_accounts,
                deny_qos,
                allow_qos,
                priority_tier,
                preempt_mode,
                is_default,
                preempt_exempt_time,
            } => {
                {
                    let mut partitions = self.partitions.write();
                    let set_as_default = *is_default == Some(true);
                    if let Some(part) = partitions.iter_mut().find(|p| p.name == *name) {
                        if let Some(n) = new_hostlist {
                            part.nodes = n.clone();
                        }
                        if let Some(sel) = selector {
                            part.selector = sel.clone();
                        }
                        if let Some(s) = state {
                            part.state = match s.to_uppercase().as_str() {
                                "UP" => spur_core::partition::PartitionState::Up,
                                "DOWN" => spur_core::partition::PartitionState::Down,
                                "DRAIN" => spur_core::partition::PartitionState::Drain,
                                _ => spur_core::partition::PartitionState::Inactive,
                            };
                        }
                        if let Some(mt) = max_time_minutes {
                            part.max_time_minutes = *mt;
                        }
                        if let Some(dt) = default_time_minutes {
                            part.default_time_minutes = *dt;
                        }
                        if let Some(mn) = max_nodes {
                            part.max_nodes = *mn;
                        }
                        if let Some(mn) = min_nodes {
                            part.min_nodes = *mn;
                        }
                        if let Some(aa) = allow_accounts {
                            part.allow_accounts = aa.clone();
                        }
                        if let Some(ag) = allow_groups {
                            part.allow_groups = ag.clone();
                        }
                        if let Some(da) = deny_accounts {
                            part.deny_accounts = da.clone();
                        }
                        if let Some(dq) = deny_qos {
                            part.deny_qos = dq.clone();
                        }
                        if let Some(aq) = allow_qos {
                            part.allow_qos = aq.clone();
                        }
                        if let Some(pt) = priority_tier {
                            part.priority_tier = *pt;
                        }
                        if let Some(pm) = preempt_mode {
                            part.preempt_mode = match pm.to_lowercase().as_str() {
                                "cancel" => spur_core::partition::PreemptMode::Cancel,
                                "requeue" => spur_core::partition::PreemptMode::Requeue,
                                "suspend" => spur_core::partition::PreemptMode::Suspend,
                                _ => spur_core::partition::PreemptMode::Off,
                            };
                        }
                        if let Some(def) = is_default {
                            part.is_default = *def;
                        }
                        if let Some(et) = preempt_exempt_time {
                            part.preempt_exempt_time = *et;
                        }
                        info!(name, "partition updated");
                    } else {
                        warn!(name, "partition update for unknown partition, ignoring");
                    }
                    // Promote exactly one default: clear all others when this one was set.
                    if set_as_default {
                        for p in partitions.iter_mut() {
                            if p.name != *name {
                                p.is_default = false;
                            }
                        }
                    }
                }
                // Node/selector may have changed which nodes this partition covers.
                self.reconcile_partitions(&mut nodes);
            }
            WalOperation::PartitionDelete { name } => {
                {
                    let mut partitions = self.partitions.write();
                    let len_before = partitions.len();
                    partitions.retain(|p| p.name != *name);
                    if partitions.len() < len_before {
                        // Record the tombstone so this name is suppressed from the
                        // spur.conf baseline on the next restart.
                        self.deleted_partition_names.write().insert(name.clone());
                        info!(name, "partition deleted");
                    }
                }
                // A node whose partition was deleted needs to fall back to
                // another matching partition (or the cluster default).
                self.reconcile_partitions(&mut nodes);
            }
            WalOperation::ReservationCreate { reservation } => {
                let mut reservations = self.reservations.write();
                if reservations.iter().any(|r| r.name == reservation.name) {
                    warn!(
                        name = %reservation.name,
                        "duplicate reservation create in WAL apply, ignoring"
                    );
                } else {
                    reservations.push(reservation.clone());
                    response.reservation_created = true;
                    info!(name = %reservation.name, "reservation created");
                }
            }
            WalOperation::ReservationUpdate {
                name,
                duration_minutes,
                add_nodes,
                remove_nodes,
                add_users,
                remove_users,
                add_accounts,
                remove_accounts,
            } => {
                let mut reservations = self.reservations.write();
                Self::apply_reservation_update_locked(
                    reservations.as_mut(),
                    name,
                    *duration_minutes,
                    add_nodes,
                    remove_nodes,
                    add_users,
                    remove_users,
                    add_accounts,
                    remove_accounts,
                );
                info!(name, "reservation updated");
            }
            WalOperation::ReservationDelete { name } => {
                let deleted = {
                    let reservations = self.reservations.read();
                    reservations.iter().find(|r| r.name == *name).cloned()
                };
                let mut reservations = self.reservations.write();
                let len_before = reservations.len();
                reservations.retain(|r| r.name != *name);
                if reservations.len() < len_before {
                    if let Some(res) = deleted {
                        if res.flags.no_hold_jobs {
                            Self::detach_jobs_from_deleted_reservation_jobs(&mut jobs, name);
                        } else {
                            Self::hold_jobs_for_deleted_reservation_jobs(&mut jobs, name);
                        }
                        info!(name, "reservation deleted");
                    }
                }
            }

            // Native k0s cluster operations. All idempotent/replay-safe: NodeK0sAssign is
            // keyed by node name, token insert/revoke are keyed by id, phase is a last-write set.
            WalOperation::NodeK0sAssign {
                name,
                role,
                mesh_ip,
                pod_cidr,
            } => {
                // Reuses the `nodes` write guard from the top of this fn.
                if let Some(node) = nodes.get_mut(name) {
                    if node.k0s_role != Some(*role) {
                        if let Some(old) = node.k0s_role {
                            self.k0s_role_counts.dec(old);
                        }
                        self.k0s_role_counts.inc(*role);
                    }
                    node.k0s_role = Some(*role);
                    node.k0s_mesh_ip = Some(mesh_ip.clone());
                    node.k0s_pod_cidr = Some(pod_cidr.clone());
                }
            }
            WalOperation::NodeK0sClear { name } => {
                if let Some(node) = nodes.get_mut(name) {
                    if let Some(old) = node.k0s_role.take() {
                        self.k0s_role_counts.dec(old);
                    }
                    node.k0s_mesh_ip = None;
                    node.k0s_pod_cidr = None;
                    node.k0s_last_error = None;
                }
            }
            WalOperation::NodeK0sSetError { name, error } => {
                if let Some(node) = nodes.get_mut(name) {
                    node.k0s_last_error = error.clone();
                }
            }
            WalOperation::K0sMemberNodesAdd { nodes: add } => {
                let mut k0s = self.k0s.write();
                k0s.member_nodes = crate::cluster_k8s::merge_member_nodes(&k0s.member_nodes, add);
            }
            WalOperation::K0sMemberNodesRemove { nodes: drop } => {
                let mut k0s = self.k0s.write();
                k0s.member_nodes =
                    crate::cluster_k8s::subtract_member_nodes(&k0s.member_nodes, drop);
            }
            WalOperation::K0sSetPhase {
                phase,
                control_plane_node,
                control_plane_nodes,
                member_nodes,
                reset_requested,
            } => {
                let mut k0s = self.k0s.write();
                k0s.phase = *phase;
                k0s.reset_requested = *reset_requested;
                // Teardown resets cluster identity so the next `up` starts clean (no stale scope/CP);
                // the branches are exclusive so a non-empty field passed alongside Down can't be set
                // then immediately wiped.
                if *phase == spur_core::k0s::K0sPhase::Down {
                    k0s.member_nodes.clear();
                    k0s.control_plane_node = None;
                    k0s.control_plane_nodes.clear();
                } else {
                    if control_plane_node.is_some() {
                        k0s.control_plane_node = control_plane_node.clone();
                    }
                    if !control_plane_nodes.is_empty() {
                        k0s.control_plane_nodes = control_plane_nodes.clone();
                    }
                    if !member_nodes.is_empty() {
                        k0s.member_nodes = member_nodes.clone();
                    }
                }
            }
            WalOperation::EvictTerminalJobs { job_ids } => {
                // Re-check finalized: spare an id requeued between propose and
                // apply. Deterministic — every replica applies in the same order.
                let evicted: HashSet<JobId> = job_ids
                    .iter()
                    .filter(|id| jobs.get(id).is_some_and(|j| j.state.is_finalized()))
                    .copied()
                    .collect();
                if !evicted.is_empty() {
                    jobs.retain(|id, _| !evicted.contains(id));
                    self.steps
                        .write()
                        .retain(|_, s| !evicted.contains(&s.job_id));
                }
            }
        }
        self.next_job_id.store(next_id, Ordering::Relaxed);
        response
    }
}

/// Snapshot data for Raft serialization.
/// Must include all durable cluster state so a follower can fully restore from it.
#[derive(serde::Serialize, serde::Deserialize)]
struct ClusterSnapshot {
    jobs: Vec<Job>,
    nodes: Vec<Node>,
    reservations: Vec<Reservation>,
    /// The leader's authoritative partition table. `None` = pre-partition-support
    /// snapshot (field absent) → restore falls back to the config baseline.
    /// `Some(_)`, empty included, is installed verbatim so a leader with zero
    /// partitions is not reseeded from a follower's local config.
    #[serde(default)]
    partitions: Option<Vec<Partition>>,
    /// Names of partitions deleted at runtime. Suppresses config-file
    /// partitions with the same name from re-seeding on restart.
    #[serde(default)]
    deleted_partition_names: HashSet<String>,
    steps: Vec<JobStep>,
    license_pool: HashMap<String, u64>,
    #[serde(default)]
    tokens: Vec<spur_core::admission::AdmissionToken>,
    /// Configured BB total (immutable; serialized for observability but, like
    /// `license_pool`, NOT restored — config stays authoritative). Per-job
    /// staging phase rides along on each `Job`.
    #[serde(default)]
    burst_buffer_total_gb: u64,
    /// cluster-wide k0s state (phase, control-plane node, join-token metadata). Unlike
    /// license_pool/burst_buffer this is runtime-authoritative allocated state and MUST be
    /// restored (see restore_from_snapshot).
    #[serde(default)]
    k0s: spur_core::k0s::K0sClusterState,
    /// Job-id high-water mark. Eviction removes the high-id tail, so restoring
    /// from survivors alone would reissue used ids; restore takes max(rebuilt, this).
    #[serde(default)]
    next_job_id: JobId,
}

impl ClusterManager {
    /// Apply features/weight from the first matching NodeConfig, reverting to
    /// node defaults when none matches so stale policy from a previously matching
    /// entry does not persist.
    fn apply_node_config_policy(&self, node: &mut Node) {
        for nc in self.config().nodes.iter() {
            if node_config_matches(nc, &node.name, &node.labels) {
                node.features = nc.features.clone();
                node.weight = nc.weight;
                if node.address.is_none() {
                    if let Some(ref cfg_addr) = nc.address {
                        node.address = Some(cfg_addr.clone());
                    }
                }
                return;
            }
        }
        node.reset_config_policy();
    }

    /// Re-evaluate partition membership and NodeConfig policy (features, weight)
    /// for all nodes against the current config. Called after snapshot restore to
    /// handle config changes that occurred between snapshot creation and restart.
    fn reconcile_partitions(&self, nodes: &mut HashMap<String, Node>) {
        let partitions = self.partitions.read();
        for node in nodes.values_mut() {
            let mut matched = Vec::new();
            for part in partitions.iter() {
                if partition_matches_node(part, &node.name, &node.labels) {
                    matched.push(part.name.clone());
                }
            }
            if matched.is_empty() {
                if let Some(dp) = partitions.iter().find(|p| p.is_default) {
                    matched.push(dp.name.clone());
                } else if let Some(first) = partitions.first() {
                    matched.push(first.name.clone());
                }
            }
            node.partitions = matched;

            self.apply_node_config_policy(node);
        }
    }
}

impl StateMachineApply for ClusterManager {
    fn apply_operation(&self, op: &WalOperation) -> ClientResponse {
        self.apply_operation(op)
    }

    fn snapshot_state(&self) -> Result<Vec<u8>, anyhow::Error> {
        let snap = ClusterSnapshot {
            jobs: self.jobs.read().values().cloned().collect(),
            nodes: self.nodes.read().values().cloned().collect(),
            reservations: self.reservations.read().clone(),
            partitions: Some(self.partitions.read().clone()),
            deleted_partition_names: self.deleted_partition_names.read().clone(),
            steps: self.steps.read().values().cloned().collect(),
            license_pool: self.license_pool.read().clone(),
            tokens: self.tokens.read().values().cloned().collect(),
            burst_buffer_total_gb: *self.burst_buffer_total_gb.read(),
            k0s: self.k0s.read().clone(),
            next_job_id: self.next_job_id.load(Ordering::Relaxed),
        };
        serde_json::to_vec(&snap).map_err(Into::into)
    }

    fn restore_from_snapshot(&self, data: &[u8]) -> Result<(), anyhow::Error> {
        let snap = serde_json::from_slice::<ClusterSnapshot>(data)?;

        // Fold in the persisted high-water mark so evicting the high-id tail
        // can't lower next_job_id and reissue used ids (absent → 0, harmless).
        let mut next_id = self.config().controller.first_job_id.max(snap.next_job_id);
        let mut jobs = self.jobs.write();
        jobs.clear();
        for job in snap.jobs {
            next_id = next_id.max(job.job_id + 1);
            jobs.insert(job.job_id, job);
        }

        let mut nodes = self.nodes.write();
        nodes.clear();
        for node in snap.nodes {
            nodes.insert(node.name.clone(), node);
        }
        self.k0s_role_counts
            .recompute_from(nodes.values().map(|n| n.k0s_role));

        *self.reservations.write() = snap.reservations;

        // Restore tombstone set first — used below to filter the config baseline.
        *self.deleted_partition_names.write() = snap.deleted_partition_names.clone();

        // `snap.partitions` is the leader's authoritative set, installed
        // wholesale. Only a pre-partition snapshot (`None`) falls back to the
        // config baseline; an authoritative empty set installs verbatim.
        {
            let mut partitions = self.partitions.write();
            *partitions = match snap.partitions {
                Some(p) => p,
                None => {
                    let mut base = self.config().build_partitions();
                    base.retain(|p| !snap.deleted_partition_names.contains(&p.name));
                    base
                }
            };
        }
        // The installed table is authoritative; no config seed remains for the
        // tail log to override.
        self.config_seeded_partitions.write().clear();

        let mut steps = self.steps.write();
        steps.clear();
        for step in snap.steps {
            steps.insert((step.job_id, step.step_id), step);
        }

        // license_pool is the configured total (immutable); it is intentionally
        // NOT restored from the snapshot so config stays authoritative and any
        // historical drift in old snapshots is discarded. Availability is
        // derived from the restored jobs. burst_buffer_total_gb follows the
        // same rule; per-job BB staging phase rides along on each restored Job.

        let mut tokens = self.tokens.write();
        tokens.clear();
        for token in snap.tokens {
            tokens.insert(token.id.clone(), token);
        }

        // k0s phase + join-token metadata are runtime-authoritative allocated state
        // (NOT config-derived like license_pool/burst_buffer) — restore them.
        *self.k0s.write() = snap.k0s;

        // Must stay inside the `jobs` write guard: a reader seeing the cleared
        // map with this watermark would treat every live job as reclaimable.
        self.next_job_id.store(next_id, Ordering::Relaxed);

        // Re-evaluate partition membership and NodeConfig policy
        // for all nodes against the current config.
        self.reconcile_partitions(&mut nodes);

        info!(
            jobs = jobs.len(),
            nodes = nodes.len(),
            "restored cluster state from Raft snapshot"
        );
        Ok(())
    }
}

fn job_candidate_node_names(job: &Job, nodes: &[spur_core::node::Node]) -> Vec<String> {
    let placement = spur_sched::node_match::NodePlacement::new(job);
    let required = spur_sched::backfill::job_resource_request(job);

    nodes
        .iter()
        .filter(|node| {
            placement.allows_name(&node.name)
                && placement.in_partition(node)
                && placement.has_features(node)
                && node.total_resources.can_satisfy(&required)
        })
        .map(|n| n.name.clone())
        .collect()
}

fn reservation_fence_reason(
    job: &Job,
    cluster_state: &spur_sched::traits::ClusterState,
) -> Option<PendingReason> {
    let placement = spur_sched::node_match::NodePlacement::new(job);
    let candidates = job_candidate_node_names(job, cluster_state.nodes);
    if candidates.is_empty() {
        return None;
    }

    let now = Utc::now();
    let duration = job
        .spec
        .time_limit
        .unwrap_or(spur_sched::UNLIMITED_JOB_DURATION);
    let mut maint_block = false;
    let mut blocked = 0;
    let mut listed_blocked = false;
    let mut unblocked = 0;

    for node_name in &candidates {
        let mut node_blocked = false;
        for res in cluster_state.reservations {
            if reservation::prospective_overlap(job, res, node_name, now, duration) {
                node_blocked = true;
                if res.flags.maint {
                    maint_block = true;
                }
            }
        }
        if node_blocked {
            blocked += 1;
            listed_blocked |= placement.nodelist_is_additive() && placement.is_listed(node_name);
        } else {
            unblocked += 1;
        }
    }

    let needed = (job.spec.num_nodes as usize).max(1);
    if blocked == 0 || (!listed_blocked && unblocked >= needed) {
        return None;
    }
    if maint_block {
        Some(PendingReason::ReservedMaintenance)
    } else {
        Some(PendingReason::ReqNodeNotAvail)
    }
}

fn retain_unblocked(
    candidates: &mut Vec<PendingJobCandidate>,
    reason_updates: &mut Vec<(JobId, PendingReason)>,
    mut block: impl FnMut(&Job) -> Option<PendingReason>,
) {
    candidates.retain(|candidate| {
        let Some(reason) = block(&candidate.job) else {
            return true;
        };
        record_pending_reason_update(reason_updates, candidate, reason);
        false
    });
}

/// Outcome of an eligibility gate for one candidate. Exhaustive so every gate
/// branch must name an outcome: a silent drop is a compile error.
enum GateOutcome {
    /// Keep the candidate in the schedulable set.
    Keep,
    /// Drop and queue `reason` when the candidate permits reason updates.
    Block(PendingReason),
    /// Drop without recording a reason. Only the burst-buffer stage-in handoff
    /// uses this: the gate reserved capacity and enqueued the job for staging,
    /// so it is tagged `BurstBufferStageIn` on the next cycle. Do not use this
    /// to bypass tagging in any other gate.
    DropUntagged,
}

/// Like [`retain_unblocked`], but skips not-yet-eligible candidates and lets
/// `gate` reserve in-pass headroom on the keep path. A [`GateOutcome::Block`]
/// drop queues its reason when the candidate permits reason updates;
/// [`GateOutcome::DropUntagged`] drops without a reason.
fn retain_eligible(
    candidates: &mut Vec<PendingJobCandidate>,
    reason_updates: &mut Vec<(JobId, PendingReason)>,
    mut gate: impl FnMut(&mut Job) -> GateOutcome,
) {
    candidates.retain_mut(|candidate| {
        if !candidate.scheduling_eligible {
            return true;
        }
        match gate(&mut candidate.job) {
            GateOutcome::Keep => true,
            GateOutcome::Block(reason) => {
                record_pending_reason_update(reason_updates, candidate, reason);
                false
            }
            GateOutcome::DropUntagged => false,
        }
    });
}

fn record_pending_reason_update(
    reason_updates: &mut Vec<(JobId, PendingReason)>,
    candidate: &PendingJobCandidate,
    reason: PendingReason,
) {
    if candidate.tag_reason {
        reason_updates.push((candidate.job.job_id, reason));
    }
}

/// `Reservation` if the job's `--reservation` is absent/inactive/expired or
/// denies it, else `None`.
fn reservation_block(
    job: &Job,
    reservations: &[Reservation],
    now: chrono::DateTime<Utc>,
) -> Option<spur_core::job::PendingReason> {
    use spur_core::job::PendingReason;
    let res_name = job.spec.reservation.as_ref()?;
    if res_name.is_empty() {
        return None;
    }
    match reservations.iter().find(|r| r.name == *res_name) {
        Some(r)
            if r.is_active(now) && r.allows_user(&job.spec.user, job.spec.account.as_deref()) =>
        {
            None
        }
        _ => Some(PendingReason::Reservation),
    }
}

/// QOS priority is baked into the job's base priority at submit time, so it
/// seeds the multiplicative formula directly rather than being a flat delta.
fn compute_effective_priority(
    base_priority: u32,
    fair_share: f64,
    age_minutes: i64,
    partition_tier: u32,
) -> u32 {
    spur_sched::priority::effective_priority(base_priority, fair_share, age_minutes, partition_tier)
}

/// Reason a job is ineligible because currently-available licenses cannot satisfy
/// its `license:` GRES requests, or `None`. Reported as `Licenses`. `available`
/// is the configured total minus licenses held by active jobs.
fn license_block(job: &Job, pool: &HashMap<String, u64>) -> Option<spur_core::job::PendingReason> {
    use spur_core::job::PendingReason;
    let lic_req = extract_license_requirements(&job.spec);
    for (lic, count) in &lic_req {
        if pool.get(lic).copied().unwrap_or(0) < *count {
            return Some(PendingReason::Licenses);
        }
    }
    None
}

/// `Err(reason)` if the job would exceed a QOS group/per-user cap. `reserved`
/// folds in headroom claimed earlier this pass so it can't over-subscribe. On
/// success, extends `job.preferred_nodes` with any nodes credited toward the
/// grp-node cap so placement actually lands on one of them, and returns the
/// node count actually charged against `grp_tres` so the caller can record it
/// (rather than the job's raw `num_nodes`) in `reserved`.
fn qos_block_with(
    job: &mut Job,
    qos: &Qos,
    jobs: &HashMap<JobId, Job>,
    nodes: &HashMap<String, Node>,
    reserved: &PassReservations,
    consumed_wall_minutes: Option<u64>,
) -> Result<u64, spur_core::job::PendingReason> {
    let Some(qos_name) = job.spec.qos.as_ref() else {
        return Ok(0);
    };
    let user = &job.spec.user;
    let mut running_count = jobs
        .values()
        .filter(|j| {
            j.state == JobState::Running
                && j.spec.user == *user
                && j.spec.qos.as_deref() == Some(qos_name.as_str())
        })
        .count() as u32;
    // Count only earlier-submitted jobs (lower job_id) so a later job never
    // makes an earlier, within-limit job retroactively blocked.
    let submitted_count = jobs
        .values()
        .filter(|j| {
            j.job_id < job.job_id
                && (j.state == JobState::Pending || j.state == JobState::Running)
                && j.spec.user == *user
                && j.spec.qos.as_deref() == Some(qos_name.as_str())
        })
        .count() as u32;
    let mut user_running_tres = sum_running_tres(jobs, |j| {
        j.spec.user == *user && j.spec.qos.as_deref() == Some(qos_name.as_str())
    });
    let mut qos_running_tres =
        sum_running_tres(jobs, |j| j.spec.qos.as_deref() == Some(qos_name.as_str()));

    let user_key = (user.clone(), qos_name.clone());
    running_count += reserved.qos_user_count.get(&user_key).copied().unwrap_or(0);
    if let Some(t) = reserved.qos_user_tres.get(&user_key) {
        user_running_tres.add(t);
    }
    if let Some(t) = reserved.qos_grp.get(qos_name) {
        qos_running_tres.add(t);
    }

    let already_claimed = reserved.qos_claimed_nodes.get(qos_name);
    let qos_occupied: HashSet<String> =
        occupied_nodes(jobs, |j| j.spec.qos.as_deref() == Some(qos_name.as_str()))
            .into_iter()
            .filter(|n| !already_claimed.is_some_and(|claimed| claimed.contains(n)))
            .collect();
    let (grp_node_charge, reusable_nodes) = new_distinct_nodes_needed(job, &qos_occupied, nodes);

    match check_qos_limits_with_grp_node_charge(
        job,
        qos,
        running_count,
        submitted_count,
        &user_running_tres,
        &qos_running_tres,
        consumed_wall_minutes,
        grp_node_charge,
    ) {
        QosCheckResult::Allowed => {
            job.preferred_nodes.extend(reusable_nodes);
            Ok(grp_node_charge)
        }
        QosCheckResult::Blocked(reason) => Err(reason),
    }
}

/// `Err(reason)` if the job would exceed its (user, account) association cap (no
/// account means unconstrained). `reserved` folds in this pass's claimed headroom.
/// On success, extends `job.preferred_nodes` with any nodes credited toward the
/// grp-node cap so placement actually lands on one of them, and returns the node
/// count actually charged against `grp_tres` so the caller can record it (rather
/// than the job's raw `num_nodes`) in `reserved`.
fn account_block_with(
    job: &mut Job,
    assoc_cache: &AssociationCache,
    jobs: &HashMap<JobId, Job>,
    nodes: &HashMap<String, Node>,
    reserved: &PassReservations,
) -> Result<u64, spur_core::job::PendingReason> {
    let Some(account) = job.spec.account.as_deref().filter(|a| !a.is_empty()) else {
        return Ok(0);
    };
    let user = &job.spec.user;
    let limits = assoc_cache.limits(user, account);

    let mut running_count = jobs
        .values()
        .filter(|j| {
            j.state == JobState::Running
                && j.spec.user == *user
                && j.spec.account.as_deref() == Some(account)
        })
        .count() as u32;
    // Count only earlier-submitted jobs (lower job_id) so a later job never
    // makes an earlier, within-limit job retroactively blocked.
    let submitted_count = jobs
        .values()
        .filter(|j| {
            j.job_id < job.job_id
                && (j.state == JobState::Pending || j.state == JobState::Running)
                && j.spec.user == *user
                && j.spec.account.as_deref() == Some(account)
        })
        .count() as u32;
    let mut account_running_tres =
        sum_running_tres(jobs, |j| j.spec.account.as_deref() == Some(account));

    running_count += reserved
        .account_user_count
        .get(&(user.clone(), account.to_string()))
        .copied()
        .unwrap_or(0);
    if let Some(t) = reserved.account_grp.get(account) {
        account_running_tres.add(t);
    }

    let already_claimed = reserved.account_claimed_nodes.get(account);
    let account_occupied: HashSet<String> =
        occupied_nodes(jobs, |j| j.spec.account.as_deref() == Some(account))
            .into_iter()
            .filter(|n| !already_claimed.is_some_and(|claimed| claimed.contains(n)))
            .collect();
    let (grp_node_charge, reusable_nodes) =
        new_distinct_nodes_needed(job, &account_occupied, nodes);

    match check_account_limits_with_grp_node_charge(
        job,
        &limits,
        running_count,
        submitted_count,
        &account_running_tres,
        grp_node_charge,
    ) {
        AccountCheckResult::Allowed => {
            job.preferred_nodes.extend(reusable_nodes);
            Ok(grp_node_charge)
        }
        AccountCheckResult::Blocked(reason) => Err(reason),
    }
}

/// For a partition OR-list, returns `None` when any requested partition is Up
/// and permits the request. When every Up alternative rejects it, returns that
/// block reason (`PartitionTimeLimit` for a wall-time cap, else
/// `PartitionConfig`); unknown names return `PartitionConfig` and all-inactive
/// alternatives return `PartitionInactive`.
fn partition_block(job: &Job, partitions: &[Partition]) -> Option<spur_core::job::PendingReason> {
    use spur_core::job::PendingReason;
    use spur_core::partition::PartitionState;

    let partition_spec = job
        .spec
        .partition
        .as_deref()
        .filter(|spec| !spec.is_empty())?;
    let requested = requested_partition_names(Some(partition_spec))
        .map(|name| partitions.iter().find(|part| part.name == name))
        .collect::<Option<Vec<_>>>();
    let Some(requested) = requested else {
        return Some(PendingReason::PartitionConfig);
    };
    if requested.is_empty() {
        return Some(PendingReason::PartitionConfig);
    }

    let mut has_up_partition = false;
    let mut block_reason: Option<PendingReason> = None;
    for part in requested {
        if part.state != PartitionState::Up {
            continue;
        }
        has_up_partition = true;
        match partition_limit_block(job, part) {
            None => return None,
            Some(reason) => {
                block_reason.get_or_insert(reason);
            }
        }
    }

    Some(if has_up_partition {
        block_reason.unwrap_or(PendingReason::PartitionConfig)
    } else {
        PendingReason::PartitionInactive
    })
}

/// Reject a user-requested wall-time exceeding a requested partition's `MaxTime`,
/// per `EnforcePartLimits` (`All` = all must fit, `Any` = at least one).
fn validate_partition_time_limit(
    spec: &JobSpec,
    mode: EnforcePartLimits,
    partitions: &[Partition],
) -> Result<(), SubmitError> {
    if mode == EnforcePartLimits::No {
        return Ok(());
    }
    let Some(time_limit) = spec.time_limit.as_ref() else {
        return Ok(());
    };
    let Some(partition_spec) = spec.partition.as_deref().filter(|p| !p.is_empty()) else {
        return Ok(());
    };

    // Check every requested partition regardless of state (Slurm parity), so the
    // outcome doesn't depend on transient Down/Drain. Existence checked upstream.
    let requested = spur_core::partition::matched_partitions(Some(partition_spec), partitions);
    if requested.is_empty() {
        return Ok(());
    }

    let allows = |part: &&Partition| partition_time_allows(part, Some(time_limit));
    let satisfied = if mode == EnforcePartLimits::All {
        requested.iter().all(allows)
    } else {
        requested.iter().any(allows)
    };
    if satisfied {
        return Ok(());
    }

    let offending = requested
        .iter()
        .find(|p| !allows(p))
        .unwrap_or(&requested[0]);
    let requested_str =
        spur_core::config::format_time_seconds(Some(time_limit.num_seconds().max(0)));
    let max_str = spur_core::config::format_time_seconds(
        offending.max_time_minutes.map(|m| i64::from(m) * 60),
    );
    Err(SubmitError::invalid(format!(
        "Requested time limit is invalid (missing or exceeds some limit): \
         {requested_str} exceeds MaxTime {max_str} of partition '{}'",
        offending.name
    )))
}

/// True when a job's wall-time fits the partition's `MaxTime`. `None` `MaxTime`
/// (UNLIMITED) or `None` time limit always fits. Compared at second precision so
/// a request just over the cap is not truncated down to an allowed minute.
fn partition_time_allows(part: &Partition, time_limit: Option<&chrono::Duration>) -> bool {
    match (part.max_time_minutes, time_limit) {
        (Some(max_mins), Some(tl)) => tl.num_seconds() <= i64::from(max_mins) * 60,
        _ => true,
    }
}

/// The reason a partition cannot ever schedule this job (static limits), or
/// `None` if it can. Node-count violations report `PartitionConfig`; wall-time
/// violations report the more specific `PartitionTimeLimit` (Slurm parity).
fn partition_limit_block(job: &Job, part: &Partition) -> Option<spur_core::job::PendingReason> {
    use spur_core::job::PendingReason;
    if let Some(max) = part.max_nodes {
        if job.spec.num_nodes > max {
            return Some(PendingReason::PartitionConfig);
        }
    }
    if part.min_nodes > 0 && job.spec.num_nodes < part.min_nodes {
        return Some(PendingReason::PartitionConfig);
    }
    if !partition_time_allows(part, job.spec.time_limit.as_ref()) {
        return Some(PendingReason::PartitionTimeLimit);
    }
    None
}

fn sum_running_tres(jobs: &HashMap<JobId, Job>, pred: impl Fn(&Job) -> bool) -> TresRecord {
    let mut tres = TresRecord::new();
    // Node TRES is the count of distinct nodes occupied (union of `allocated_nodes`), not summed
    // `num_nodes` — a node packed by two jobs counts once. CPU/Memory/GPU stay additive.
    let mut distinct_nodes: HashSet<&str> = HashSet::new();
    let mut unplaced_nodes: u64 = 0;
    for j in jobs.values() {
        if j.state != JobState::Running || !pred(j) {
            continue;
        }
        tres.add(&job_tres(j));
        if j.allocated_nodes.is_empty() {
            // Running job without recorded placement (transient/edge): fall back
            // to its requested count so the limit is never under-counted.
            unplaced_nodes += j.spec.num_nodes as u64;
        } else {
            distinct_nodes.extend(j.allocated_nodes.iter().map(String::as_str));
        }
    }
    // `job_tres` set Node to the summed `num_nodes` above; replace it with the
    // real distinct-node occupancy.
    tres.set(TresType::Node, distinct_nodes.len() as u64 + unplaced_nodes);
    tres
}

/// The limits-versus-usage snapshot behind `scontrol show assoc_mgr`. Usage is
/// measured the way the admission gate measures it, so a record reads as the
/// scheduler sees it rather than as a second opinion.
pub(crate) struct AssocMgrInfo {
    pub qos_records: Vec<ScopeLimitUsage>,
    pub assoc_records: Vec<ScopeLimitUsage>,
    pub limits_readable: bool,
}

/// Whether the caps in an assoc-mgr response can be trusted as complete.
/// Evaluated per cache: a cache is only ever unreadable when accounting is
/// enabled yet its snapshot has not loaded. With accounting off there are no
/// caps to load, so a cold cache is not a fault — otherwise the banner reads as
/// a standing fault on a cluster that simply has no limits.
pub(crate) fn limits_readable(enabled: bool, qos_loaded: bool, assoc_loaded: bool) -> bool {
    !enabled || (qos_loaded && assoc_loaded)
}

/// A scope name only counts when it is actually set: an empty QOS or account
/// string is the same as none.
fn named(value: Option<&str>) -> Option<&str> {
    value.filter(|v| !v.is_empty())
}

fn qos_of(job: &Job) -> Option<&str> {
    named(job.spec.qos.as_deref())
}

fn account_of(job: &Job) -> Option<&str> {
    named(job.spec.account.as_deref())
}

fn is_submitted(job: &Job) -> bool {
    matches!(job.state, JobState::Pending | JobState::Running)
}

/// Scope names carried by queued work. Folded in alongside the defined scopes so
/// a QOS deleted after its jobs queued, or one named while the cache was cold, is
/// still accounted for instead of silently dropped.
fn queued_scopes(
    jobs: &HashMap<JobId, Job>,
    scope_of: impl Fn(&Job) -> Option<&str>,
) -> BTreeSet<String> {
    jobs.values()
        .filter(|j| is_submitted(j))
        .filter_map(|j| scope_of(j).map(str::to_owned))
        .collect()
}

/// One scope's consumption from the job table, with an entry per user holding
/// work in it. `defined_users` are users the accounting definitions know of; they
/// get an entry at zero usage so a cap on an idle association is still reported.
/// Group figures are the scope's own and are never narrowed by `only_user`, since
/// a group cap cannot be judged from one user's share.
///
/// `None` when `only_user` is set and that user neither holds work nor is defined
/// under the scope: a user-scoped read only sees scopes it takes part in, so the
/// rest are dropped rather than leaked as cluster-wide inventory.
fn scope_usage(
    jobs: &HashMap<JobId, Job>,
    scope: &str,
    scope_of: impl Fn(&Job) -> Option<&str>,
    only_user: Option<&str>,
    defined_users: &[String],
    caps_for: impl Fn(&str) -> PerUserCaps,
) -> Option<ScopeLimitUsage> {
    let in_scope = |j: &Job| scope_of(j) == Some(scope);

    let mut users: BTreeSet<&str> = jobs
        .values()
        .filter(|j| is_submitted(j) && in_scope(j))
        .map(|j| j.spec.user.as_str())
        .collect();
    users.extend(defined_users.iter().map(String::as_str));
    users.retain(|u| only_user.is_none() || only_user == Some(u));

    if only_user.is_some() && users.is_empty() {
        return None;
    }

    Some(ScopeLimitUsage {
        scope: scope.to_owned(),
        grp_running_jobs: jobs
            .values()
            .filter(|j| j.state == JobState::Running && in_scope(j))
            .count() as u32,
        grp_submitted_jobs: jobs
            .values()
            .filter(|j| is_submitted(j) && in_scope(j))
            .count() as u32,
        grp_running_tres: sum_running_tres(jobs, in_scope),
        users: users
            .into_iter()
            .map(|user| {
                let mine = |j: &Job| in_scope(j) && j.spec.user == user;
                UserLimitUsage {
                    user: user.to_owned(),
                    running_jobs: jobs
                        .values()
                        .filter(|j| j.state == JobState::Running && mine(j))
                        .count() as u32,
                    submitted_jobs: jobs.values().filter(|j| is_submitted(j) && mine(j)).count()
                        as u32,
                    running_tres: sum_running_tres(jobs, mine),
                    caps: caps_for(user),
                }
            })
            .collect(),
        ..Default::default()
    })
}

/// Distinct nodes currently occupied by running jobs matching `pred`, so the
/// group-node check can let a pending job of the same group pack onto them
/// instead of always requiring brand-new distinct nodes.
fn occupied_nodes(jobs: &HashMap<JobId, Job>, pred: impl Fn(&Job) -> bool) -> HashSet<String> {
    jobs.values()
        .filter(|j| j.state == JobState::Running && pred(j))
        .flat_map(|j| j.allocated_nodes.iter().cloned())
        .collect()
}

/// Distinct NEW nodes `job` would need beyond `occupied`, given each occupied
/// node's live spare capacity and scheduling eligibility, plus the specific
/// occupied node names that qualified as reusable (for the caller to feed to
/// `Job::preferred_nodes`, so placement actually lands on one of them instead
/// of the admission credit going unenforced). A job that fits entirely on
/// already-occupied nodes needs 0 new nodes.
///
/// Uses `NodePlacement::new_ignoring_preferred_nodes`, not `NodePlacement::new`:
/// this is called once for the account gate and once for the QOS gate on the
/// SAME job, and the first gate to run may already have written its own
/// credited nodes into `job.preferred_nodes` before the second one runs. If
/// this function's own placement view honored that field, the first gate's
/// credit would leak in as a spurious nodelist restriction on the second
/// gate's independent reuse computation — incorrectly narrowing (or, if the
/// first gate's credit alone already covered `num_nodes`, entirely excluding)
/// nodes the second gate's own dimension legitimately occupies.
///
/// A candidate must also be genuinely schedulable (`Node::is_schedulable`)
/// and not claimed by the managed k0s cluster (`Node::is_k0s_reserved`) to be
/// counted reusable — the same gates real backfill placement applies via
/// `NodePlacement::matches`. Without them, a Down/Draining/k0s-reserved node
/// that still hosts a running job with spare capacity would be credited here,
/// admission would treat the pending job as covered, but real placement can
/// never land it there — turning the credit into a standing nodelist bias
/// that starves the job instead of merely costing it a pending cycle.
///
/// Reservation-fencing is intentionally not checked here (matching
/// `job_candidate_node_names`'s scope): worst case a reserved node is
/// optimistically counted as reusable, the job clears this gate, and the real
/// backfill placement step still refuses to place it there — no
/// oversubscription risk, just a possible extra pending cycle (unlike the
/// schedulability/k0s gates above, a reservation can end and free the node up
/// again, so this can't wedge a job indefinitely). Exclusive jobs never reuse
/// an already-occupied node, since they require the whole node to themselves.
///
/// Known soft-cap limitation: when a job only partially reuses (charge > 0),
/// its `preferred_nodes` bias is additive, not restrictive (see
/// `NodePlacement`), so backfill is free to place its remaining new node(s)
/// anywhere. If the specific occupied node credited here becomes unavailable
/// between this admission check and actual placement — e.g. claimed by a job
/// admitted in a later pass — the job can end up needing one more genuinely
/// new distinct node than it was charged for. `qos_claimed_nodes`/
/// `account_claimed_nodes` on `PassReservations` close this gap within a
/// single pass; across passes it remains a best-effort accounting, bounded
/// in practice by real placement being the final authority on node choice.
fn new_distinct_nodes_needed(
    job: &Job,
    occupied: &HashSet<String>,
    nodes: &HashMap<String, Node>,
) -> (u64, HashSet<String>) {
    if job.spec.exclusive {
        return (job.spec.num_nodes as u64, HashSet::new());
    }
    let placement = spur_sched::node_match::NodePlacement::new_ignoring_preferred_nodes(job);
    let required = spur_sched::backfill::job_resource_request(job);
    let reusable: HashSet<String> = occupied
        .iter()
        .filter(|name| {
            nodes.get(*name).is_some_and(|node| {
                placement.allows_name(&node.name)
                    && placement.in_partition(node)
                    && placement.has_features(node)
                    && node.is_schedulable()
                    && !node.is_k0s_reserved()
                    && node.can_satisfy_request(&required)
            })
        })
        .cloned()
        .collect();
    let reusable_count = reusable.len() as u32;
    let charge = job
        .spec
        .num_nodes
        .saturating_sub(reusable_count.min(job.spec.num_nodes)) as u64;
    (charge, reusable)
}

/// The TRES a single job occupies (its scheduling footprint). Shared by the
/// running-usage sum and the same-pass reservation accounting so both count a
/// job's CPU/node/mem/GPU footprint identically.
fn job_tres(job: &Job) -> TresRecord {
    let mut tres = TresRecord::new();
    tres.set(
        TresType::Cpu,
        (job.spec.num_tasks * job.spec.cpus_per_task) as u64,
    );
    tres.set(TresType::Node, job.spec.num_nodes as u64);
    tres.set(
        TresType::Memory,
        effective_memory_mb(&job.spec, job.spec.num_nodes),
    );
    tres.set(TresType::Gpu, effective_gpus(&job.spec, job.spec.num_nodes));
    tres
}

/// Headroom claimed by jobs already kept earlier in a single `pending_jobs()`
/// pass, folded into the running aggregates so aggregate QOS/account caps
/// (`grp_tres`, per-user TRES, per-user running-job counts) can't be
/// over-subscribed within one pass — the same guarantee licenses and
/// burst-buffer capacity already get via priority-ordered reservation.
///
/// `{qos,account}_claimed_nodes` track which already-occupied nodes were
/// credited toward a kept job's grp-node charge this pass, so a later
/// candidate can't independently claim the same node's spare capacity too —
/// only one job can actually land on it. Without this, two same-pass
/// candidates can both compute a reuse credit for one node, both get
/// admitted, and the grp-node cap is exceeded once real placement resolves
/// which of them actually gets it.
#[derive(Default)]
struct PassReservations {
    qos_grp: HashMap<String, TresRecord>,
    qos_user_tres: HashMap<(String, String), TresRecord>,
    qos_user_count: HashMap<(String, String), u32>,
    qos_claimed_nodes: HashMap<String, HashSet<String>>,
    account_grp: HashMap<String, TresRecord>,
    account_user_count: HashMap<(String, String), u32>,
    account_claimed_nodes: HashMap<String, HashSet<String>>,
}

impl PassReservations {
    /// Record that `job` (kept this pass) now occupies its footprint against the
    /// QOS and account aggregates, so lower-priority jobs later in the pass see
    /// the reduced headroom. `qos_node_charge`/`account_node_charge` are the
    /// grp-node counts `qos_block_with`/`account_block_with` actually charged
    /// this job (which may be less than its raw `num_nodes` when it packed onto
    /// already-occupied nodes) — the grp aggregates' Node dimension reflects
    /// that charge, not the raw request, so it stays consistent with what
    /// admission actually granted. Per-user TRES keeps the raw node count,
    /// since per-job/per-user caps always bound a job's own footprint.
    ///
    /// `qos_claimed_nodes`/`account_claimed_nodes` are both populated from the
    /// job's full (possibly QOS-and-account-merged) `preferred_nodes`, not
    /// just the calling dimension's own credited subset. This is
    /// intentionally conservative: the node's spare capacity is a single
    /// physical resource this job may consume regardless of which dimension
    /// motivated crediting it, so ANY later same-pass candidate — QOS or
    /// account — must treat it as spoken for. Splitting this per dimension
    /// would let two same-pass candidates of different QOS/account both
    /// claim the same shared node's same spare capacity as long as they
    /// approached it from different dimensions, reopening the double-credit
    /// gap this whole mechanism exists to close.
    fn reserve(&mut self, job: &Job, qos_node_charge: u64, account_node_charge: u64) {
        let tres = job_tres(job);
        let user = job.spec.user.clone();
        if let Some(qos) = job.spec.qos.as_ref().filter(|q| !q.is_empty()) {
            let mut qos_grp_tres = tres.clone();
            qos_grp_tres.set(TresType::Node, qos_node_charge);
            self.qos_grp
                .entry(qos.clone())
                .or_default()
                .add(&qos_grp_tres);
            let key = (user.clone(), qos.clone());
            self.qos_user_tres
                .entry(key.clone())
                .or_default()
                .add(&tres);
            *self.qos_user_count.entry(key).or_insert(0) += 1;
            self.qos_claimed_nodes
                .entry(qos.clone())
                .or_default()
                .extend(job.preferred_nodes.iter().cloned());
        }
        if let Some(account) = job.spec.account.as_deref().filter(|a| !a.is_empty()) {
            let account = account.to_string();
            let mut account_grp_tres = tres.clone();
            account_grp_tres.set(TresType::Node, account_node_charge);
            self.account_grp
                .entry(account.clone())
                .or_default()
                .add(&account_grp_tres);
            *self
                .account_user_count
                .entry((user, account.clone()))
                .or_insert(0) += 1;
            self.account_claimed_nodes
                .entry(account)
                .or_default()
                .extend(job.preferred_nodes.iter().cloned());
        }
    }
}

/// Burst-buffer capacity (GB) a job's `--bb` string reserves cluster-wide.
/// Shares the grammar with the agent's stage wrapper via `spur_core`.
fn extract_bb_requirement(spec: &JobSpec) -> u64 {
    spec.burst_buffer
        .as_deref()
        .map(spur_core::burst_buffer::parse_capacity_gb)
        .unwrap_or(0)
}

/// `BurstBufferResources` if the job needs more BB capacity than is currently
/// free, else `None`. Reported when an absolute shortage means the job can
/// never stage in the current cluster state. `free_gb` is the configured total
/// minus capacity reserved by staging/active jobs.
fn burst_buffer_block(job: &Job, free_gb: u64) -> Option<spur_core::job::PendingReason> {
    use spur_core::job::PendingReason;
    // A job that already reserved capacity (Staging/Ready) is not blocked on
    // resources — it is either staging or dispatchable.
    if job.bb_stage_state != BbStageState::None {
        return None;
    }
    let req = extract_bb_requirement(&job.spec);
    if req > 0 && req > free_gb {
        Some(PendingReason::BurstBufferResources)
    } else {
        None
    }
}

fn extract_license_requirements(spec: &JobSpec) -> HashMap<String, u64> {
    let mut licenses = HashMap::new();
    for gres in &spec.gres {
        if let Some((name, ltype, count)) = spur_core::resource::parse_gres(gres) {
            if name == "license" {
                let lic_name = ltype.unwrap_or_else(|| "unknown".to_string());
                *licenses.entry(lic_name).or_insert(0) += count as u64;
            }
        }
    }
    licenses
}

#[derive(Debug, PartialEq)]
pub(crate) enum RegistrationAction {
    Skip,
    Update,
    Register,
}

pub(crate) fn evaluate_registration(
    existing: Option<&Node>,
    incoming_resources: &ResourceSet,
) -> RegistrationAction {
    match existing {
        None => RegistrationAction::Register,
        Some(node) if node.total_resources != *incoming_resources => RegistrationAction::Update,
        Some(_) => RegistrationAction::Skip,
    }
}

/// Returns true if a node matches a partition's membership criteria.
/// Match occurs if the node satisfies EITHER the hostlist OR the label selector.
pub(crate) fn partition_matches_node(
    partition: &spur_core::partition::Partition,
    node_name: &str,
    labels: &HashMap<String, String>,
) -> bool {
    let matches_selector = !partition.selector.is_empty()
        && partition
            .selector
            .iter()
            .all(|(k, v)| labels.get(k) == Some(v));

    let matches_hostlist = if partition.nodes.is_empty() {
        false
    } else if partition.nodes.eq_ignore_ascii_case("ALL") {
        true
    } else {
        spur_core::hostlist::expand(&partition.nodes)
            .map(|hosts| hosts.iter().any(|h| h == node_name))
            .unwrap_or(false)
    };

    matches_selector || matches_hostlist
}

/// Returns true if a NodeConfig entry applies to a node (by hostname pattern OR
/// label selector).
pub(crate) fn node_config_matches(
    nc: &spur_core::config::NodeConfig,
    node_name: &str,
    labels: &HashMap<String, String>,
) -> bool {
    let matches_names = if nc.names.is_empty() {
        false
    } else if nc.names.eq_ignore_ascii_case("ALL") {
        true
    } else {
        spur_core::hostlist::expand(&nc.names)
            .map(|hosts| hosts.iter().any(|h| h == node_name))
            .unwrap_or(false)
    };

    let matches_selector =
        !nc.selector.is_empty() && nc.selector.iter().all(|(k, v)| labels.get(k) == Some(v));

    matches_names || matches_selector
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkDownPolicy {
    Allowed,
    Suppressed,
}

/// Withholds DOWN marking for `grace` after leadership is first observed: a
/// non-leader records no heartbeats, so every `last_heartbeat` is outage-stale.
pub struct LeadershipGrace {
    grace: std::time::Duration,
    leader_since: Option<std::time::Instant>,
}

impl LeadershipGrace {
    pub fn new(grace: std::time::Duration) -> Self {
        Self {
            grace,
            leader_since: None,
        }
    }

    pub fn observe(&mut self, is_leader: bool, now: std::time::Instant) -> MarkDownPolicy {
        if !is_leader {
            self.leader_since = None;
            return MarkDownPolicy::Suppressed;
        }
        let since = *self.leader_since.get_or_insert(now);
        if now.saturating_duration_since(since) >= self.grace {
            MarkDownPolicy::Allowed
        } else {
            MarkDownPolicy::Suppressed
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum HealthAction {
    MarkDown {
        name: String,
        old_state: NodeState,
        admin_locked: bool,
    },
    Recover {
        name: String,
        old_state: NodeState,
    },
}

/// Attribution captured when an admin sets a node's state. A set-time is
/// recorded only when a reason is present; clearing a reason (e.g. resume to
/// Idle) leaves both unset. Captured at the call site, not in apply, so Raft
/// replay stays deterministic.
fn reason_attribution(
    reason: &Option<String>,
    reason_uid: Option<u32>,
) -> (Option<u32>, Option<DateTime<Utc>>) {
    match reason {
        Some(_) => (reason_uid, Some(Utc::now())),
        None => (None, None),
    }
}

pub(crate) fn recovered_node_state(node: Option<&Node>) -> NodeState {
    let Some(node) = node else {
        return NodeState::Idle;
    };
    // The node may have changed state since the action was computed, so re-derive
    // from allocation only while recovery is still a valid transition for it.
    if node
        .state
        .transition(&NodeEvent::HeartbeatRecovered, node.admin_locked)
        .is_none()
    {
        return node.state;
    }
    let mut recovered = node.clone();
    // Clear the transient failure state so allocation state can be recomputed.
    recovered.state = NodeState::Idle;
    recovered.update_state_from_alloc();
    recovered.state
}

pub(crate) fn evaluate_node_health(
    nodes: &[&Node],
    now: DateTime<Utc>,
    timeout_secs: u64,
) -> Vec<HealthAction> {
    let threshold = chrono::Duration::seconds(timeout_secs as i64);
    let mut actions = Vec::new();

    for node in nodes {
        let Some(hb) = node.last_heartbeat else {
            continue;
        };
        let stale = now - hb > threshold;

        if stale {
            if node
                .state
                .transition(&NodeEvent::HeartbeatTimeout, node.admin_locked)
                .is_some()
            {
                actions.push(HealthAction::MarkDown {
                    name: node.name.clone(),
                    old_state: node.state,
                    admin_locked: node.admin_locked,
                });
            }
        } else if node
            .state
            .transition(&NodeEvent::HeartbeatRecovered, node.admin_locked)
            .is_some()
        {
            actions.push(HealthAction::Recover {
                name: node.name.clone(),
                old_state: node.state,
            });
        }
    }
    actions
}

fn apply_default_partition(spec: &mut JobSpec, partitions: &[Partition]) {
    if spec.partition.as_deref().is_some_and(|p| p.is_empty()) {
        spec.partition = None;
    }
    if spec.partition.is_none() {
        if let Some(default_part) = partitions.iter().find(|p| p.is_default) {
            spec.partition = Some(default_part.name.clone());
        } else if let Some(first) = partitions.first() {
            spec.partition = Some(first.name.clone());
        }
    }
}

/// The partitions a job's wall-time defaulting has to satisfy. A job may list
/// several (e.g. "gpu,cpu"); fall back to the default/first partition when the
/// field is unset or names nothing known.
fn partitions_for_defaulting<'a>(
    spec: &JobSpec,
    partitions: &'a [Partition],
) -> Vec<&'a Partition> {
    let matched = spec
        .partition
        .as_deref()
        .map(|p| spur_core::partition::matched_partitions(Some(p), partitions))
        .unwrap_or_default();
    if !matched.is_empty() {
        return matched;
    }
    partitions
        .iter()
        .find(|p| p.is_default)
        .or_else(|| partitions.first())
        .into_iter()
        .collect()
}

/// The per-job wall-clock ceilings a submission is already subject to, in minutes.
/// `None` means no figure applies — the limit is unset, or its cache has not loaded
/// yet, in which case defaulting proceeds as if the limit did not exist rather than
/// holding the submission. A ceiling of `0` is the "block all" sentinel, not a
/// zero-length budget, so it is never a value to default to.
#[derive(Clone, Copy, Default)]
pub(crate) struct WallCaps {
    qos_minutes: Option<u32>,
    account_minutes: Option<u32>,
}

impl WallCaps {
    fn usable(minutes: Option<u32>) -> Option<u32> {
        minutes.filter(|m| *m > 0)
    }
}

/// Fill in a job's wall-time when none was requested. Each requested partition
/// resolves to `DefaultTime`, else (only when `cluster_default_minutes > 0`) its
/// `MaxTime`, else the cluster fallback. The smallest resulting limit is chosen
/// so the default fits every requested partition; if any requested partition
/// resolves to no bound the governing QOS's or account's `MaxWall` supplies the
/// limit instead, and a job with neither stays unbounded (unchanged upgrade
/// behavior).
fn apply_default_time_limit(
    spec: &mut JobSpec,
    partitions: &[Partition],
    cluster_default_minutes: u32,
    caps: WallCaps,
) {
    if spec.time_limit.is_some() {
        return;
    }

    let requested = partitions_for_defaulting(spec, partitions);
    if requested.is_empty() {
        return;
    }

    let resolve = |part: &Partition| -> Option<u32> {
        if let Some(minutes) = part.default_time_minutes {
            return Some(minutes);
        }
        // Opt-in reclamation: only bound the job when a cluster fallback is set.
        if cluster_default_minutes == 0 {
            return None;
        }
        Some(part.max_time_minutes.unwrap_or(cluster_default_minutes))
    };

    let mut chosen: Option<u32> = None;
    for part in &requested {
        match resolve(part) {
            // A requested partition that imposes no default leaves the job's wall
            // ceiling as the only source of one.
            None => {
                chosen = wall_cap_default(&requested, caps);
                break;
            }
            Some(minutes) => chosen = Some(chosen.map_or(minutes, |c| c.min(minutes))),
        }
    }
    if let Some(minutes) = chosen {
        spec.time_limit = Some(chrono::Duration::minutes(i64::from(minutes)));
    }
}

/// The per-job wall ceiling as a default. A job has to satisfy the QOS's and the
/// account's `MaxWall` both, so the smaller of the two is the limit it is really
/// held to. That value is then kept under the requested partitions' `MaxTime`,
/// above which the job would pend forever on a limit it never asked for — where
/// today it simply runs unbounded.
fn wall_cap_default(requested: &[&Partition], caps: WallCaps) -> Option<u32> {
    let cap = [
        WallCaps::usable(caps.qos_minutes),
        WallCaps::usable(caps.account_minutes),
    ]
    .into_iter()
    .flatten()
    .min()?;

    // The limit has to stay schedulable somewhere: a requested partition with no
    // `MaxTime` takes any, otherwise the most permissive one is the ceiling. With
    // no partition in play nothing caps the ceiling, so the wall cap stands.
    let partition_ceiling = if requested.iter().any(|p| p.max_time_minutes.is_none()) {
        None
    } else {
        requested.iter().filter_map(|p| p.max_time_minutes).max()
    };

    [Some(cap), partition_ceiling].into_iter().flatten().min()
}

/// Resolve the submitting user's default account from the association cache
/// when `--account` was not provided (mirrors `apply_default_partition`).
fn apply_default_account(spec: &mut JobSpec, assoc_cache: &AssociationCache) {
    if !assoc_cache.is_loaded() {
        return;
    }
    if spec.account.as_deref().is_some_and(|a| !a.is_empty()) {
        return;
    }
    let (account, ..) = assoc_cache.resolve(&spec.user, None);
    if let Some(acct) = account.filter(|a| !a.is_empty()) {
        spec.account = Some(acct);
    }
}

/// Reject a client-supplied account that isn't a real association, or — with `require_association`
/// set — no account at all. Fails closed on a cold cache when accounting is enabled (an unverified
/// free label otherwise), so a not-yet-loaded cache can't wave a spoofed account through.
fn validate_user_account(
    spec: &JobSpec,
    assoc_cache: &AssociationCache,
    accounting: &spur_core::config::AccountingConfig,
) -> Result<(), SubmitError> {
    let Some(account) = spec.account.as_deref().filter(|a| !a.is_empty()) else {
        if accounting.require_association {
            // Cold cache: the user's default account may still resolve once the refresh
            // loop populates associations, so this is transient (retryable), not permanent.
            if accounting.enabled() && !assoc_cache.is_loaded() {
                return Err(SubmitError::unavailable(format!(
                    "account associations are temporarily unavailable (accounting cache not loaded); \
                     no default account resolved for user '{}' yet. Try again once the accounting \
                     database is reachable.",
                    spec.user
                )));
            }
            let hint = if assoc_cache.is_loaded() {
                ""
            } else {
                QOS_ACCOUNTING_HINT
            };
            return Err(SubmitError::invalid(format!(
                "no account resolved for user '{}' (no --account given and no default account on file){hint}. \
                 Specify --account explicitly, or contact your cluster admin to set a default: \
                 sacctmgr modify user name={} set defaultaccount=<account>",
                spec.user, spec.user
            )));
        }
        return Ok(());
    };
    match assoc_cache.account_membership(&spec.user, account) {
        AccountMembership::Member => Ok(()),
        AccountMembership::CacheUnavailable if !accounting.enabled() => Ok(()),
        AccountMembership::CacheUnavailable => {
            // Operator-visible alarm: the fence is meant to hold but we cannot read associations.
            warn!(
                user = %spec.user,
                account,
                "association cache not loaded while accounting is enabled — denying account-scoped \
                 submission (fail closed). Check the accounting database is reachable."
            );
            // Transient: the refresh loop populates the cache shortly after start, so
            // this must be retryable (Unavailable), not a permanent rejection.
            Err(SubmitError::unavailable(format!(
                "account associations are temporarily unavailable (accounting cache not loaded); \
                 cannot verify user '{}' is associated with account '{account}'. Try again once \
                 the accounting database is reachable.",
                spec.user
            )))
        }
        AccountMembership::NotMember(valid_accounts) if valid_accounts.is_empty() => {
            Err(SubmitError::invalid(format!(
                "user '{}' has no account associations. Contact your cluster admin to run: sacctmgr add user name={} account=<account>",
                spec.user, spec.user
            )))
        }
        AccountMembership::NotMember(valid_accounts) => Err(SubmitError::invalid(format!(
            "user '{}' is not associated with account '{account}'. Accounts associated with this user: [{}].",
            spec.user,
            valid_accounts.join(", ")
        ))),
    }
}

const QOS_ACCOUNTING_HINT: &str =
    " (hint: accounting may not be enabled on this controller -- check controller logs)";

/// Resolve a job's QOS at submit, in Slurm's order: explicit `--qos` (must
/// exist and be permitted for the association) → association default →
/// cluster fallback (`accounting.default_qos`, also gated by the
/// association's allow-list) → reject if `accounting.require_qos`, else
/// accept with no QOS.
fn apply_default_qos(
    spec: &mut JobSpec,
    assoc_cache: &AssociationCache,
    qos_cache: &QosCache,
    accounting: &spur_core::config::AccountingConfig,
) -> Result<(), SubmitError> {
    let given_account = spec.account.as_deref().filter(|a| !a.is_empty());
    // Resolved once and shared by both branches below: `resolve()` reads
    // the account, default QOS, and allow-list under a single lock, so a
    // concurrent cache refresh can't validate one against the other's
    // stale snapshot.
    let (account, default_qos, allowed_qos) = assoc_cache.resolve(&spec.user, given_account);
    // A stale pinned default (removed from qos_cache since it was set) is
    // treated as unset for authorization, matching how the fallback chain
    // below already ignores it rather than letting it block every explicit
    // `--qos` a user could otherwise submit.
    let default_qos_for_auth = default_qos
        .as_deref()
        .filter(|d| qos_cache.get(d).is_some())
        .map(str::to_owned);

    if let Some(name) = spec.qos.as_deref().filter(|n| !n.is_empty()) {
        if qos_cache.get(name).is_none() {
            let hint = if qos_cache.is_loaded() {
                String::new()
            } else {
                QOS_ACCOUNTING_HINT.into()
            };
            return Err(SubmitError::invalid(format!(
                "QOS '{name}' does not exist{hint}"
            )));
        }
        if !qos_permitted(&allowed_qos, default_qos_for_auth.as_deref(), name) {
            return Err(SubmitError::invalid(format!(
                "QOS '{name}' is not permitted for user '{}' under account '{}'",
                spec.user,
                account.as_deref().unwrap_or_default()
            )));
        }
        return Ok(());
    }

    if let Some(default_qos) = default_qos {
        if qos_cache.get(&default_qos).is_some() {
            spec.qos = Some(default_qos);
            return Ok(());
        }
        warn!(
            user = %spec.user,
            account = account.as_deref().unwrap_or_default(),
            qos = %default_qos,
            "association default QOS no longer exists, ignoring"
        );
    }

    // A configured fallback naming a nonexistent QOS is a hard error — silently
    // ignoring it would leave the job unenforced, the gap this closes. A fallback
    // that exists but isn't permitted for this association degrades like a stale
    // default instead, since it reflects the association's own restriction, not
    // a misconfiguration.
    let fallback = accounting.default_qos.trim();
    if !fallback.is_empty() {
        if qos_cache.get(fallback).is_none() {
            let hint = if qos_cache.is_loaded() {
                String::new()
            } else {
                QOS_ACCOUNTING_HINT.into()
            };
            return Err(SubmitError::invalid(format!(
                "configured default QOS '{fallback}' does not exist{hint}"
            )));
        }
        if qos_permitted(&allowed_qos, default_qos_for_auth.as_deref(), fallback) {
            spec.qos = Some(fallback.to_string());
            return Ok(());
        }
        warn!(
            user = %spec.user,
            account = account.as_deref().unwrap_or_default(),
            qos = %fallback,
            "cluster default QOS not permitted for this association, ignoring"
        );
    }

    if accounting.require_qos {
        return Err(SubmitError::invalid(
            "no QOS specified and no default QOS is configured for this user/account",
        ));
    }

    Ok(())
}

/// Expand a job spec into one or more submittable specs. For non-array jobs,
/// returns the spec unchanged. For array jobs, returns N task specs with
/// array metadata populated and `array_spec` cleared.
fn expand_job_specs(spec: JobSpec, parent_job_id: JobId) -> anyhow::Result<Vec<JobSpec>> {
    let Some(ref array_spec_str) = spec.array_spec else {
        return Ok(vec![spec]);
    };

    let array = spur_core::array::parse_array_spec(array_spec_str)
        .map_err(|e| anyhow::anyhow!("invalid array spec: {}", e))?;

    let max_concurrent = if array.max_concurrent > 0 {
        Some(array.max_concurrent)
    } else {
        None
    };

    Ok(array
        .task_ids
        .iter()
        .map(|&task_id| {
            let mut task_spec = spec.clone();
            task_spec.array_spec = None;
            task_spec.array_job_id = Some(parent_job_id);
            task_spec.array_task_id = Some(task_id);
            task_spec.array_max_concurrent = max_concurrent;
            task_spec
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use spur_core::job::JobSpec;
    use spur_core::resource::{ResourceAllocations, ResourceSet};
    use spur_metrics::job::JobMetricsSnapshot;
    use tempfile::TempDir;

    #[test]
    fn submission_size_accepts_normal_script() {
        let spec = JobSpec {
            script: Some("#!/bin/bash\necho hello\n".into()),
            ..Default::default()
        };
        assert!(check_submission_size(&spec).is_ok());
    }

    #[test]
    fn submission_size_rejects_oversized_script() {
        let spec = JobSpec {
            script: Some("x".repeat(MAX_JOB_SPEC_SIZE + 1)),
            ..Default::default()
        };
        let err = check_submission_size(&spec).expect_err("oversized script must be rejected");
        assert!(matches!(err, SubmitError::InvalidArgument(_)));
        assert!(err.to_string().contains("too large"));
    }

    #[test]
    fn submission_size_counts_all_fields() {
        // Split the payload across environment and argv (not script) to prove the
        // check bounds the whole serialized spec, not just the script field.
        let big = "y".repeat(MAX_JOB_SPEC_SIZE / 2 + 1024);
        let spec = JobSpec {
            environment: std::iter::once(("BIG".to_string(), big.clone())).collect(),
            argv: vec![big],
            ..Default::default()
        };
        let err =
            check_submission_size(&spec).expect_err("env + argv over the cap must be rejected");
        assert!(matches!(err, SubmitError::InvalidArgument(_)));
    }

    fn test_config() -> SlurmConfig {
        SlurmConfig {
            cluster_name: "test".into(),
            controller: spur_core::config::ControllerConfig {
                first_job_id: 1,
                ..Default::default()
            },
            accounting: Default::default(),
            scheduler: Default::default(),
            auth: Default::default(),
            partitions: vec![spur_core::config::PartitionConfig {
                name: "default".into(),
                default: true,
                state: "UP".into(),
                nodes: "ALL".into(),
                selector: Default::default(),
                max_time: None,
                default_time: None,
                max_nodes: None,
                min_nodes: 1,
                allow_accounts: Vec::new(),
                allow_groups: Vec::new(),
                deny_accounts: Vec::new(),
                deny_qos: Vec::new(),
                allow_qos: Vec::new(),
                priority_tier: 1,
                preempt_mode: String::new(),
                preempt_exempt_time: None,
            }],
            nodes: Vec::new(),
            network: Default::default(),
            logging: Default::default(),
            kubernetes: Default::default(),
            cluster: Default::default(),
            notifications: Default::default(),
            power: Default::default(),
            federation: Default::default(),
            topology: None,
            isolation: Default::default(),
            licenses: HashMap::new(),
            burst_buffer: Default::default(),
            update: Default::default(),
            metrics: Default::default(),
            rest_api: Default::default(),
            hooks: Default::default(),
            devices: Default::default(),
            admission: Default::default(),
            rlimits: Default::default(),
            cgroup: Default::default(),
            mpi: Default::default(),
            health: Default::default(),
        }
    }

    async fn test_cluster(dir: &TempDir) -> Arc<ClusterManager> {
        test_cluster_with_config(dir, test_config()).await
    }

    async fn test_cluster_with_config(dir: &TempDir, config: SlurmConfig) -> Arc<ClusterManager> {
        let cm = Arc::new(ClusterManager::new(config, dir.path()).unwrap());
        let handle = crate::raft::start_raft(1, &["[::1]:0".into()], dir.path(), cm.clone())
            .await
            .unwrap();
        // Wait for the single-node Raft to self-elect before returning.
        // Without this, the first propose() call may hit a not-yet-leader
        // node and silently fail.
        handle
            .raft
            .wait(Some(std::time::Duration::from_secs(5)))
            .metrics(|m| m.current_leader == Some(1), "leader elected")
            .await
            .expect("single-node raft did not self-elect within 5s");
        cm.set_raft(handle.raft);
        cm
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn interactive_reap_candidates_reaps_only_idle_allocations() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let t0 = Utc::now();

        // First sighting seeds each allocation's timer; nothing is reaped yet.
        assert!(cm.interactive_reap_candidates(&[1, 2], t0, 60).is_empty());

        // Job 2 pings recently; job 1 stays silent past the limit.
        cm.record_job_keepalive_at(2, t0 + chrono::Duration::seconds(55));
        let later = t0 + chrono::Duration::seconds(61);
        assert_eq!(cm.interactive_reap_candidates(&[1, 2], later, 60), vec![1]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn interactive_reap_candidates_disabled_only_prunes() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let t0 = Utc::now();

        // A stale entry exists, but with the limit disabled it is pruned (the
        // allocation is no longer running) rather than reaped.
        cm.record_job_keepalive_at(1, t0);
        assert!(cm.interactive_reap_candidates(&[], t0, 0).is_empty());
        // The prune actually happened: job 1's timer is gone, not just skipped
        // by the disabled early return.
        assert!(
            cm.interactive_last_seen.read().is_empty(),
            "pruning must drop entries for allocations no longer running"
        );

        // Because the old entry was pruned, job 1 reappearing reseeds to `now`
        // and is not immediately reaped despite the original stale timestamp.
        let later = t0 + chrono::Duration::seconds(3600);
        assert!(cm.interactive_reap_candidates(&[1], later, 60).is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reset_interactive_last_seen_reseeds_on_leadership_regain() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let t0 = Utc::now();

        // A prior stint's timestamp for a still-running allocation, frozen at
        // t0 because a follower records no keepalives.
        cm.record_job_keepalive_at(1, t0);

        // On regain the stale entry is present, so seed-to-now can't correct
        // it and the live allocation looks idle past the limit.
        let later = t0 + chrono::Duration::seconds(3600);
        assert_eq!(
            cm.interactive_reap_candidates(&[1], later, 60),
            vec![1],
            "a stale carried-over entry would reap a live allocation"
        );

        // The reaper's follower -> leader reset clears the map, so the next
        // check reseeds job 1 to `now` and exempts it.
        cm.reset_interactive_last_seen();
        assert!(
            cm.interactive_reap_candidates(&[1], later, 60).is_empty(),
            "after reset, a live allocation is reseeded and not reaped"
        );
    }

    fn basic_spec(name: &str) -> JobSpec {
        JobSpec {
            name: name.into(),
            user: "testuser".into(),
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            work_dir: "/tmp".into(),
            ..Default::default()
        }
    }

    fn srun_spec(name: &str) -> JobSpec {
        let mut spec = basic_spec(name);
        spec.srun_job = true;
        spec
    }

    /// Assert a submission was denied and the rejection carries the exact Slurm
    /// reason `code` in parentheses, alongside the human sentence.
    fn assert_submit_denied(err: &SubmitError, code: &str) {
        let SubmitError::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains(&format!("({code})")), "got: {msg}");
    }

    /// Build a cluster backed by a real spur.conf on disk so `reconfigure()`
    /// has a path to re-read. Returns the cluster and the conf path so tests
    /// can rewrite the file and reconcile.
    async fn test_cluster_with_conf_file(
        dir: &TempDir,
        toml: &str,
    ) -> (Arc<ClusterManager>, PathBuf) {
        let conf_path = dir.path().join("spur.conf");
        std::fs::write(&conf_path, toml).unwrap();
        let config = SlurmConfig::load_from_file(&conf_path).unwrap();
        let cm = Arc::new(
            ClusterManager::new_with_config_path(config, dir.path(), Some(conf_path.clone()))
                .unwrap(),
        );
        let handle = crate::raft::start_raft(1, &["[::1]:0".into()], dir.path(), cm.clone())
            .await
            .unwrap();
        handle
            .raft
            .wait(Some(std::time::Duration::from_secs(5)))
            .metrics(|m| m.current_leader == Some(1), "leader elected")
            .await
            .expect("single-node raft did not self-elect within 5s");
        cm.set_raft(handle.raft);
        (cm, conf_path)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dispatch_cooldown_marks_then_expires_and_respects_disable() {
        let dir = TempDir::new().unwrap();

        // Enabled (default 30s): a cooled node is reported until it expires.
        let cm = test_cluster_with_config(&dir, test_config()).await;
        register_node(&cm, "worker1", 8, 16000);
        register_node(&cm, "worker2", 8, 16000);
        assert!(cm.nodes_on_dispatch_cooldown().is_empty());
        cm.cool_down_node("worker1");
        assert!(cm.nodes_on_dispatch_cooldown().contains("worker1"));

        // The scheduler's node view excludes the cooled node, keeps the other.
        let names: HashSet<String> = cm
            .nodes_off_dispatch_cooldown(&[])
            .into_iter()
            .map(|n| n.name)
            .collect();
        assert!(
            !names.contains("worker1"),
            "cooled node excluded from scheduling"
        );
        assert!(names.contains("worker2"), "healthy node still schedulable");

        // A past instant is pruned on read, so an expired cooldown clears.
        cm.node_dispatch_cooldowns.write().insert(
            "worker1".into(),
            std::time::Instant::now() - std::time::Duration::from_secs(1),
        );
        assert!(!cm.nodes_on_dispatch_cooldown().contains("worker1"));
        assert!(
            cm.nodes_off_dispatch_cooldown(&[])
                .iter()
                .any(|n| n.name == "worker1"),
            "expired cooldown makes the node schedulable again"
        );

        // Disabled (0s): cool_down_node is a no-op.
        let mut cfg = test_config();
        cfg.controller.dispatch_reject_cooldown_secs = 0;
        let cm0 = Arc::new(ClusterManager::new(cfg, dir.path()).unwrap());
        cm0.cool_down_node("worker1");
        assert!(cm0.nodes_on_dispatch_cooldown().is_empty());
    }

    /// GATE: a job's `--nodelist` pin is its only possible placement — cooling
    /// that node down must not starve it for the whole cooldown window, or a
    /// preempt-then-redispatch race against the same node (chronic preemption)
    /// can never make progress within a job's own retry budget.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nodes_off_dispatch_cooldown_exempts_a_cooling_node_pinned_by_a_pending_job() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        register_node(&cm, "n2", 4, 8000);

        cm.cool_down_node("n1");
        assert!(cm.nodes_on_dispatch_cooldown().contains("n1"));

        let mut spec = basic_spec("pinned");
        spec.nodelist = Some("n1".into());
        let job_id = submit_and_wait(&cm, spec);
        let pinned_job = cm.get_job(job_id).unwrap();

        let names: HashSet<String> = cm
            .nodes_off_dispatch_cooldown(std::slice::from_ref(&pinned_job))
            .into_iter()
            .map(|n| n.name)
            .collect();
        assert!(
            names.contains("n1"),
            "n1 is the pinned job's only candidate — must stay schedulable despite cooling"
        );

        // No pending job names the cooling node: back to excluded, as normal.
        let names: HashSet<String> = cm
            .nodes_off_dispatch_cooldown(&[])
            .into_iter()
            .map(|n| n.name)
            .collect();
        assert!(
            !names.contains("n1"),
            "n1 must stay excluded when nothing pins to it"
        );
    }

    /// Consumer-driven: `maybe_requeue` must honor the new `max_batch_requeue`
    /// after reconfigure, not just the swapped config value. A job whose
    /// `requeue_count` sits between the old and new caps is a no-op under the
    /// old cap but requeues to Pending under the new one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconfigure_max_batch_requeue_changes_consumer_behavior() {
        let conf = |cap: u32| {
            format!(
                "cluster_name = \"test\"\n\
                 [controller]\nmax_batch_requeue = {cap}\n\
                 [[partitions]]\nname = \"default\"\ndefault = true\nstate = \"UP\"\nnodes = \"ALL\"\n"
            )
        };
        let dir = TempDir::new().unwrap();
        let (cm, conf_path) = test_cluster_with_conf_file(&dir, &conf(3)).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "requeue-cap", "worker1");
        // Put the job in a terminal, requeue-eligible state (Failed → Pending is
        // a valid requeue transition and is NOT in the max-requeue hold set, so
        // over-cap is a clean no-op) with 5 attempts already recorded.
        {
            let mut jobs = cm.jobs.write();
            let job = jobs.get_mut(&job_id).unwrap();
            job.state = JobState::Failed;
            job.spec.requeue = true;
            job.requeue_count = 5;
        }

        // Cap = 3, count = 5 → over cap → maybe_requeue is a no-op (stays Failed).
        cm.maybe_requeue(job_id).unwrap();
        assert_eq!(
            cm.get_job(job_id).unwrap().state,
            JobState::Failed,
            "over-cap job must not requeue before reconfigure"
        );

        // Raise the cap past the attempt count and reconfigure.
        std::fs::write(&conf_path, conf(9)).unwrap();
        cm.reconfigure().unwrap();

        // Cap = 9, count = 5 → under cap → maybe_requeue returns it to Pending.
        cm.maybe_requeue(job_id).unwrap();
        settle(&cm, job_id, JobState::Pending);
        assert_eq!(
            cm.get_job(job_id).unwrap().state,
            JobState::Pending,
            "after reconfigure raised the cap, the consumer must requeue the job"
        );
    }

    #[tokio::test]
    async fn reconfigure_reloads_scheduler_tunables_live() {
        let dir = TempDir::new().unwrap();
        let (cm, conf_path) = test_cluster_with_conf_file(
            &dir,
            "cluster_name = \"test\"\n[scheduler]\nresv_overrun_minutes = 5\ncomplete_wait_secs = 30\n",
        )
        .await;
        assert_eq!(cm.config().scheduler.resv_overrun_minutes, 5);

        std::fs::write(
            &conf_path,
            "cluster_name = \"test\"\n[scheduler]\nresv_overrun_minutes = 45\ncomplete_wait_secs = 90\n",
        )
        .unwrap();
        cm.reconfigure().unwrap();

        let cfg = cm.config();
        assert_eq!(cfg.scheduler.resv_overrun_minutes, 45);
        assert_eq!(cfg.scheduler.complete_wait_secs, 90);
    }

    #[tokio::test]
    async fn reconfigure_reloads_hooks_and_notifications_live() {
        let dir = TempDir::new().unwrap();
        let (cm, conf_path) = test_cluster_with_conf_file(&dir, "cluster_name = \"test\"\n").await;
        assert!(cm.config().hooks.epilog_slurmctld.is_none());
        assert!(cm.config().notifications.webhook_url.is_none());

        std::fs::write(
            &conf_path,
            "cluster_name = \"test\"\n[hooks]\nepilog_slurmctld = \"/usr/bin/epi\"\n[notifications]\nwebhook_url = \"http://hook/\"\n",
        )
        .unwrap();
        cm.reconfigure().unwrap();

        let cfg = cm.config();
        assert_eq!(cfg.hooks.epilog_slurmctld.as_deref(), Some("/usr/bin/epi"));
        assert_eq!(
            cfg.notifications.webhook_url.as_deref(),
            Some("http://hook/")
        );
    }

    #[tokio::test]
    async fn reconfigure_reloads_license_pool_live() {
        let dir = TempDir::new().unwrap();
        let (cm, conf_path) =
            test_cluster_with_conf_file(&dir, "cluster_name = \"test\"\n[licenses]\nfluent = 5\n")
                .await;
        // Availability is derived from the pool total; no jobs hold licenses.
        assert_eq!(cm.available_licenses().get("fluent").copied(), Some(5));

        std::fs::write(
            &conf_path,
            "cluster_name = \"test\"\n[licenses]\nfluent = 20\ncomsol = 2\n",
        )
        .unwrap();
        cm.reconfigure().unwrap();

        let avail = cm.available_licenses();
        assert_eq!(
            avail.get("fluent").copied(),
            Some(20),
            "reconfigure must apply the new license total"
        );
        assert_eq!(avail.get("comsol").copied(), Some(2));
    }

    fn scalar_alloc(cpus: u32, memory_mb: u64) -> ResourceAllocations {
        ResourceAllocations::with_scalar(cpus, memory_mb)
    }

    fn per_node_for(
        nodes: &[&str],
        alloc: ResourceAllocations,
    ) -> HashMap<String, ResourceAllocations> {
        nodes
            .iter()
            .map(|n| ((*n).to_string(), alloc.clone()))
            .collect()
    }

    /// Spin until a Raft-proposed mutation is visible in memory.
    /// In tests, `propose()` can be called before the single-node Raft
    /// has finished its initial self-election, causing `client_write` to
    /// fail silently. This helper retries until the election completes
    /// and the mutation is applied.
    fn wait_for<F: Fn() -> bool>(label: &str, f: F) {
        for _ in 0..200 {
            if f() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("timed out waiting for: {label}");
    }

    fn register_node(cm: &ClusterManager, name: &str, cpus: u32, mem: u64) {
        cm.register_node(
            name.into(),
            name.into(),
            ResourceSet {
                cpus,
                memory_mb: mem,
                ..Default::default()
            },
            "127.0.0.1".into(),
            6818,
            String::new(),
            String::new(),
            spur_core::node::NodeSource::NativeHost,
            HashMap::new(),
            true,
        )
        .unwrap();
        let n = name.to_string();
        wait_for(&format!("node '{n}' registered"), || {
            cm.get_node(&n).is_some()
        });
    }

    fn health_check(program: &str, interval_secs: u64) -> HealthCheck {
        HealthCheck {
            program: program.into(),
            nodes: 1,
            interval_secs,
            timeout_secs: 60,
            max_wait_secs: 600,
            user: "nobody".into(),
            uid: 65534,
            gid: 65534,
        }
    }

    fn config_with_checks(checks: Vec<HealthCheck>) -> SlurmConfig {
        let mut cfg = test_config();
        cfg.health = spur_core::config::HealthConfig {
            checks,
            ..Default::default()
        };
        cfg
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn health_pass_submits_exclusive_pinned_job_when_due() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster_with_config(
            &dir,
            config_with_checks(vec![health_check("/opt/nhc.sh", 300)]),
        )
        .await;
        register_node(&cm, "n1", 8, 16000);

        cm.run_node_health_pass();
        wait_for("health job submitted", || {
            !cm.get_jobs(&JobFilter::default()).is_empty()
        });

        let jobs = cm.get_jobs(&JobFilter::default());
        assert_eq!(jobs.len(), 1, "exactly one health job");
        let spec = &jobs[0].spec;
        assert!(
            spec.name.starts_with(HEALTH_JOB_PREFIX),
            "reserved name marker, got {}",
            spec.name
        );
        assert!(spec.exclusive, "check must take the whole node");
        assert_eq!(spec.nodelist.as_deref(), Some("n1"), "pinned to the node");
        assert_eq!(spec.uid, 65534, "runs as the unprivileged default identity");
        assert_eq!(spec.argv, vec!["/opt/nhc.sh".to_string()]);
        assert_eq!(
            spec.partition.as_deref(),
            Some("default"),
            "submitted into the node's own partition so the pin resolves"
        );

        // A second pass while the first is in flight must not double-submit.
        cm.run_node_health_pass();
        assert_eq!(
            cm.get_jobs(&JobFilter::default()).len(),
            1,
            "no duplicate check while one is in flight"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn health_pass_no_checks_submits_nothing() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        cm.run_node_health_pass();
        assert!(
            cm.get_jobs(&JobFilter::default()).is_empty(),
            "no [[health.checks]] means no probing"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_rejects_reserved_health_name() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let spec = basic_spec("_spur-health.0.n1");
        let err = cm.submit_job(spec).unwrap_err();
        // The reject names the reserved prefix so the user understands why.
        let SubmitError::InvalidArgument(msg) = &err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains(HEALTH_JOB_PREFIX), "got: {msg}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn health_leaves_drained_node_alone() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster_with_config(
            &dir,
            config_with_checks(vec![health_check("/bin/true", 300)]),
        )
        .await;
        register_node(&cm, "n1", 8, 16000);

        // A drained node (operator maintenance, or a prior failed check) is
        // off-limits: no probe, so a failed node stays down for an operator.
        cm.update_node_state("n1", NodeState::Drain, Some("maintenance".into()), None)
            .unwrap();
        wait_for("n1 drained", || {
            cm.get_node("n1")
                .is_some_and(|n| matches!(n.state, NodeState::Drain | NodeState::Draining))
        });
        assert!(cm.node_off_limits_for_health(&cm.get_node("n1").unwrap()));
        cm.run_node_health_pass();
        assert!(
            cm.get_jobs(&JobFilter::default()).is_empty(),
            "must not probe a drained node"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn health_force_drain_resumes_when_node_empties() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster_with_config(
            &dir,
            config_with_checks(vec![health_check("/bin/true", 300)]),
        )
        .await;
        register_node(&cm, "n1", 8, 16000);

        // A node soft-drained to make room for a starved check (with no jobs
        // still running) must be resumed by the pass so the check can land.
        cm.update_node_state(
            "n1",
            NodeState::Drain,
            Some("health-check(pending): running overdue check /bin/true".into()),
            None,
        )
        .unwrap();
        wait_for("n1 force-drained", || {
            cm.get_node("n1")
                .is_some_and(|n| matches!(n.state, NodeState::Drain | NodeState::Draining))
        });

        cm.run_node_health_pass();
        wait_for("n1 resumed to idle", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Idle)
        });
    }

    /// Submit a health check via the pass and drive it to Running on `node`,
    /// returning its job id — the setup a completion-reaction test needs.
    async fn health_job_running_on(cm: &ClusterManager, node: &str) -> JobId {
        cm.run_node_health_pass();
        wait_for("health job submitted", || {
            !cm.get_jobs(&JobFilter::default()).is_empty()
        });
        let job_id = cm.get_jobs(&JobFilter::default())[0].job_id;
        let res = scalar_alloc(1, 1000);
        cm.start_job(
            job_id,
            vec![node.to_string()],
            res.clone(),
            per_node_for(&[node], res),
        )
        .unwrap();
        wait_for("health job running", || {
            cm.get_job(job_id)
                .is_some_and(|j| j.state == JobState::Running)
        });
        job_id
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn health_failed_job_sticky_drains_node() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster_with_config(
            &dir,
            config_with_checks(vec![health_check("/opt/nhc.sh", 300)]),
        )
        .await;
        register_node(&cm, "n1", 8, 16000);
        let job_id = health_job_running_on(&cm, "n1").await;

        // A real non-zero exit drains the node with the sticky failed reason.
        cm.react_to_health_job_finalized(job_id, JobState::Failed, 1);
        wait_for("n1 drained by failed check", || {
            cm.get_node("n1").is_some_and(|n| {
                matches!(n.state, NodeState::Drain | NodeState::Draining)
                    && n.state_reason
                        .as_deref()
                        .is_some_and(|r| r.starts_with("health-check(failed): "))
            })
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn health_passed_job_leaves_node_schedulable() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster_with_config(
            &dir,
            config_with_checks(vec![health_check("/opt/nhc.sh", 300)]),
        )
        .await;
        register_node(&cm, "n1", 8, 16000);
        let job_id = health_job_running_on(&cm, "n1").await;

        // A pass leaves the node in service — never drained.
        cm.react_to_health_job_finalized(job_id, JobState::Completed, 0);
        assert!(
            cm.get_node("n1")
                .is_some_and(|n| !matches!(n.state, NodeState::Drain | NodeState::Draining)),
            "a passing check must not drain the node"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn health_launch_failure_does_not_drain() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster_with_config(
            &dir,
            config_with_checks(vec![health_check("/opt/nhc.sh", 300)]),
        )
        .await;
        register_node(&cm, "n1", 8, 16000);
        let job_id = health_job_running_on(&cm, "n1").await;

        // Exit 127 means the program could not be run (a config/deploy error),
        // not a hardware fault — so the node must NOT be drained, or one bad
        // program path would drain the whole fleet.
        cm.react_to_health_job_finalized(job_id, JobState::Failed, 127);
        assert!(
            cm.get_node("n1")
                .is_some_and(|n| !matches!(n.state, NodeState::Drain | NodeState::Draining)),
            "a launch failure (exit 127) must not drain the node"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn health_concurrency_cap_bounds_submits() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config_with_checks(vec![health_check("/opt/nhc.sh", 300)]);
        // Cap health activity at a single node even though four are due at once.
        cfg.health.max_unavailable = "1".into();
        let cm = test_cluster_with_config(&dir, cfg).await;
        for n in ["n1", "n2", "n3", "n4"] {
            register_node(&cm, n, 8, 16000);
        }

        cm.run_node_health_pass();
        wait_for("first health job submitted", || {
            !cm.get_jobs(&JobFilter::default()).is_empty()
        });
        // Even a second pass can't exceed the cap while the first is in flight.
        cm.run_node_health_pass();
        assert_eq!(
            cm.get_jobs(&JobFilter::default()).len(),
            1,
            "max_unavailable=1 must cap health activity at one node"
        );
    }

    // A `Barrier` forces both callers to start at the same instant, so this races for real on
    // `node_registration_locks` instead of depending on tokio's task-scheduling luck.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn register_node_concurrent_non_admin_relabel_never_wins() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "racy-node", 4, 8000);

        let relabel = |cm: Arc<ClusterManager>, pool: &'static str, privileged: bool| {
            cm.register_node(
                "racy-node".into(),
                "racy-node".into(),
                ResourceSet {
                    cpus: 4,
                    memory_mb: 8000,
                    ..Default::default()
                },
                "127.0.0.1".into(),
                6818,
                String::new(),
                String::new(),
                spur_core::node::NodeSource::NativeHost,
                HashMap::from([("pool".to_string(), pool.to_string())]),
                privileged,
            )
        };

        let barrier = Arc::new(std::sync::Barrier::new(2));
        let (cm_a, barrier_a) = (cm.clone(), barrier.clone());
        let admin = tokio::task::spawn_blocking(move || {
            barrier_a.wait();
            relabel(cm_a, "prod", true)
        });
        let (cm_b, barrier_b) = (cm.clone(), barrier.clone());
        let non_admin = tokio::task::spawn_blocking(move || {
            barrier_b.wait();
            relabel(cm_b, "hacked", false)
        });

        let (admin_result, non_admin_result) = tokio::join!(admin, non_admin);
        admin_result
            .unwrap()
            .expect("the admin's relabel must succeed regardless of scheduling order");
        assert!(
            matches!(
                non_admin_result.unwrap(),
                Err(RegisterNodeError::PermissionDenied(_))
            ),
            "the non-admin's concurrent relabel must never win the race"
        );
        assert_eq!(
            cm.get_node("racy-node").unwrap().labels["pool"],
            "prod",
            "only the admin's labels may land, whichever task the scheduler ran first"
        );
    }

    /// Register a node already on the WireGuard mesh: it advertises its real `spur0` address (what
    /// `detect_node_address` reports when WireGuard is up) and a wg pubkey, so `provision_assignments`
    /// adopts that real address as its `k0s_mesh_ip` instead of allocating from a pool.
    fn register_meshed_node(cm: &ClusterManager, name: &str, mesh_addr: &str) {
        cm.register_node(
            name.into(),
            name.into(),
            ResourceSet {
                cpus: 4,
                memory_mb: 8000,
                ..Default::default()
            },
            mesh_addr.into(),
            6818,
            format!("pk-{name}"),
            String::new(),
            spur_core::node::NodeSource::NativeHost,
            HashMap::new(),
            true,
        )
        .unwrap();
        let n = name.to_string();
        let want = mesh_addr.to_string();
        wait_for(&format!("meshed node '{n}' registered"), || {
            cm.get_node(&n)
                .map(|x| x.address.as_deref() == Some(want.as_str()))
                .unwrap_or(false)
        });
    }

    fn submit_and_wait(cm: &ClusterManager, spec: JobSpec) -> JobId {
        let id = cm.submit_job(spec).unwrap().job_id;
        wait_for(&format!("job {id} applied"), || cm.get_job(id).is_some());
        id
    }

    /// Wait for a job to reach the expected state.
    /// Handles the test-only race where propose() is called before the
    /// single-node Raft has self-elected.
    fn settle(cm: &ClusterManager, job_id: JobId, expected: JobState) {
        wait_for(&format!("job {job_id} -> {expected:?}"), || {
            cm.get_job(job_id).is_some_and(|j| j.state == expected)
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_submit() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let spec = basic_spec("test-job");
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(spec.clone()),
        });

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.job_id, 1);
        assert_eq!(job.spec.name, "test-job");
        assert_eq!(job.state, JobState::Pending);
        assert!(cm.next_job_id.load(Ordering::Relaxed) >= 2);
    }

    fn write_hook_script(dir: &TempDir, body: &str) -> String {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;
        let path = dir.path().join("job_submit.sh");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "#!/bin/bash\n{body}").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_hook_rejects() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit = Some(write_hook_script(
            &dir,
            "echo 'denied by policy' >&2\nexit 1",
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let err = cm.submit_job(basic_spec("blocked")).unwrap_err();
        assert_eq!(err, SubmitError::invalid("denied by policy"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_hook_modifies_spec() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit = Some(write_hook_script(&dir, r#"echo '{"qos":"high"}'"#));
        let cm = test_cluster_with_config(&dir, config).await;
        cm.qos_cache().insert(Qos {
            name: "high".into(),
            priority: 5000,
            ..Default::default()
        });

        let id = submit_and_wait(&cm, basic_spec("modme"));
        assert_eq!(cm.get_job(id).unwrap().spec.qos.as_deref(), Some("high"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_hook_modify_to_invalid_partition_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit = Some(write_hook_script(&dir, r#"echo '{"partition":"nope"}'"#));
        let cm = test_cluster_with_config(&dir, config).await;

        let err = cm.submit_job(basic_spec("bad-part")).unwrap_err();
        assert_eq!(err, SubmitError::invalid("partition 'nope' not found"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_hook_modify_to_unknown_qos_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit = Some(write_hook_script(&dir, r#"echo '{"qos":"ghost"}'"#));
        let cm = test_cluster_with_config(&dir, config).await;

        let err = cm.submit_job(basic_spec("bad-qos")).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("job_submit hook set unknown QOS 'ghost'")
        );
    }

    // A hook-set qos must still respect the partition's own deny_qos ACL, not
    // just cluster-wide existence, or it reopens the hole partition/account
    // modify already gets re-checked for.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_hook_modify_to_denied_qos_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].deny_qos = vec!["high".into()];
        config.hooks.job_submit = Some(write_hook_script(&dir, r#"echo '{"qos":"high"}'"#));
        let cm = test_cluster_with_config(&dir, config).await;
        cm.qos_cache().insert(Qos {
            name: "high".into(),
            priority: 5000,
            ..Default::default()
        });

        let err = cm.submit_job(basic_spec("denied-qos")).unwrap_err();
        assert!(
            err.to_string().contains("denied on partition"),
            "expected a partition QoS-ACL rejection, got: {err:?}"
        );
    }

    // A hook that sets gres conflicting with an explicit --gpus request is
    // rejected at submit time, not silently deferred to schedule time.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_hook_gres_conflict_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit = Some(write_hook_script(
            &dir,
            r#"echo '{"gres":["gpu:mi300x:2"]}'"#,
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("gres-conflict");
        spec.gpus = Some(spur_core::gpu_request::GpuRequest {
            count: 4,
            gpu_type: None,
        });
        let err = cm.submit_job(spec).unwrap_err();
        assert!(
            format!("{err:?}").contains("conflicting gres"),
            "expected a gres-conflict rejection, got: {err:?}"
        );
    }

    // A hook can grow an unbounded field (comment/constraint/gres) past the
    // pre-hook size cap; the post-modify re-check must catch that.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_hook_modify_reenforces_size_cap() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        let huge_comment = "c".repeat(200 * 1024);
        config.hooks.job_submit = Some(write_hook_script(
            &dir,
            &format!(r#"echo '{{"comment":"{huge_comment}"}}'"#),
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("size-bypass");
        spec.script = Some("x".repeat(MAX_JOB_SPEC_SIZE - 100 * 1024));
        let err = cm.submit_job(spec).unwrap_err();
        assert!(
            err.to_string().contains("too large"),
            "expected a size-cap rejection, got: {err:?}"
        );
    }

    // A hook-set partition must be re-validated for node-count bounds too,
    // not just ACLs, or it reopens the pend-forever hole this check closes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_hook_modify_partition_bypasses_node_bounds_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions.push(spur_core::config::PartitionConfig {
            name: "small".into(),
            default: false,
            state: "UP".into(),
            nodes: "ALL".into(),
            selector: Default::default(),
            max_time: None,
            default_time: None,
            max_nodes: Some(1),
            min_nodes: 1,
            allow_accounts: Vec::new(),
            allow_groups: Vec::new(),
            deny_accounts: Vec::new(),
            deny_qos: Vec::new(),
            allow_qos: Vec::new(),
            priority_tier: 1,
            preempt_mode: String::new(),
            preempt_exempt_time: None,
        });
        config.hooks.job_submit = Some(write_hook_script(&dir, r#"echo '{"partition":"small"}'"#));
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("bounds-bypass");
        spec.num_nodes = 2;
        spec.num_tasks = 2;
        let err = cm.submit_job(spec).unwrap_err();
        assert!(
            err.to_string().contains("outside partition"),
            "expected a node-bounds rejection, got: {err:?}"
        );
    }

    // A hook-set QOS governs the job, so its MaxWall has to reach an untimed job
    // as the default; caps read before the hook would be the wrong QOS's.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_hook_set_qos_supplies_the_wall_default() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit = Some(write_hook_script(&dir, r#"echo '{"qos":"capped"}'"#));
        let cm = test_cluster_with_config(&dir, config).await;
        cm.qos_cache().insert(Qos {
            name: "capped".into(),
            limits: spur_core::accounting::QosLimits {
                max_wall_minutes: Some(30),
                ..Default::default()
            },
            ..Default::default()
        });

        let mut spec = basic_spec("hook-qos-wall");
        spec.time_limit = None;
        let id = submit_and_wait(&cm, spec);
        assert_eq!(
            cm.get_job(id).unwrap().spec.time_limit,
            Some(chrono::Duration::minutes(30))
        );
    }

    // The partition also decides the default, so a hook that reroutes the job
    // must be defaulted from the partition it ends up on.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_hook_set_partition_supplies_the_wall_default() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        let mut timed = config.partitions[0].clone();
        timed.name = "timed".into();
        timed.default = false;
        timed.default_time = Some("00:15:00".into());
        config.partitions.push(timed);
        config.hooks.job_submit = Some(write_hook_script(&dir, r#"echo '{"partition":"timed"}'"#));
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("hook-part-wall");
        spec.time_limit = None;
        let id = submit_and_wait(&cm, spec);
        assert_eq!(
            cm.get_job(id).unwrap().spec.time_limit,
            Some(chrono::Duration::minutes(15))
        );
    }

    // Defaulting only fills an absent limit: one the hook set itself stands,
    // even where a cap would have produced a different value.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_hook_set_time_limit_survives_defaulting() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit = Some(write_hook_script(
            &dir,
            r#"echo '{"qos":"capped","time_limit_minutes":45}'"#,
        ));
        let cm = test_cluster_with_config(&dir, config).await;
        cm.qos_cache().insert(Qos {
            name: "capped".into(),
            limits: spur_core::accounting::QosLimits {
                max_wall_minutes: Some(60),
                ..Default::default()
            },
            ..Default::default()
        });

        let mut spec = basic_spec("hook-own-limit");
        spec.time_limit = None;
        let id = submit_and_wait(&cm, spec);
        assert_eq!(
            cm.get_job(id).unwrap().spec.time_limit,
            Some(chrono::Duration::minutes(45))
        );
    }

    fn write_lua_script(dir: &TempDir, body: &str) -> String {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;
        let path = dir.path().join("job_submit.lua");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{body}").unwrap();
        // The runner refuses a group/world-writable hook; a permissive umask
        // would otherwise trip that check on the temp file.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_lua_hook_rejects() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit_lua = Some(write_lua_script(
            &dir,
            "function slurm_job_submit(j, u)\n  slurm.log_user('denied by lua')\n  return slurm.ERROR\nend",
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let err = cm.submit_job(basic_spec("blocked-lua")).unwrap_err();
        assert_eq!(err, SubmitError::invalid("denied by lua"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_lua_hook_modifies_spec() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit_lua = Some(write_lua_script(
            &dir,
            "function slurm_job_submit(j, u)\n  j.comment = 'lua-tagged'\n  return slurm.SUCCESS\nend",
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let id = submit_and_wait(&cm, basic_spec("lua-mod"));
        assert_eq!(
            cm.get_job(id).unwrap().spec.comment.as_deref(),
            Some("lua-tagged")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_lua_modify_to_invalid_partition_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit_lua = Some(write_lua_script(
            &dir,
            "function slurm_job_submit(j, u)\n  j.partition = 'nope'\n  return slurm.SUCCESS\nend",
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let err = cm.submit_job(basic_spec("lua-bad-part")).unwrap_err();
        assert_eq!(err, SubmitError::invalid("partition 'nope' not found"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_lua_modify_to_unknown_qos_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit_lua = Some(write_lua_script(
            &dir,
            "function slurm_job_submit(j, u)\n  j.qos = 'ghost'\n  return slurm.SUCCESS\nend",
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let err = cm.submit_job(basic_spec("lua-bad-qos")).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("job_submit hook set unknown QOS 'ghost'")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_lua_modify_to_denied_qos_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].deny_qos = vec!["high".into()];
        config.hooks.job_submit_lua = Some(write_lua_script(
            &dir,
            "function slurm_job_submit(j, u)\n  j.qos = 'high'\n  return slurm.SUCCESS\nend",
        ));
        let cm = test_cluster_with_config(&dir, config).await;
        cm.qos_cache().insert(Qos {
            name: "high".into(),
            priority: 5000,
            ..Default::default()
        });

        let err = cm.submit_job(basic_spec("lua-denied-qos")).unwrap_err();
        assert!(
            err.to_string().contains("denied on partition"),
            "expected a partition QoS-ACL rejection, got: {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_lua_gres_conflict_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit_lua = Some(write_lua_script(
            &dir,
            "function slurm_job_submit(j, u)\n  j.gres = {'gpu:mi300x:2'}\n  return slurm.SUCCESS\nend",
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("lua-gres-conflict");
        spec.gpus = Some(spur_core::gpu_request::GpuRequest {
            count: 4,
            gpu_type: None,
        });
        let err = cm.submit_job(spec).unwrap_err();
        assert!(
            format!("{err:?}").contains("conflicting gres"),
            "expected a gres-conflict rejection, got: {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_lua_modify_reenforces_size_cap() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        let huge_comment = "c".repeat(200 * 1024);
        config.hooks.job_submit_lua = Some(write_lua_script(
            &dir,
            &format!(
                "function slurm_job_submit(j, u)\n  j.comment = string.rep('c', {})\n  return slurm.SUCCESS\nend",
                huge_comment.len()
            ),
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("lua-size-bypass");
        spec.script = Some("x".repeat(MAX_JOB_SPEC_SIZE - 100 * 1024));
        let err = cm.submit_job(spec).unwrap_err();
        assert!(
            err.to_string().contains("too large"),
            "expected a size-cap rejection, got: {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_lua_modify_partition_bypasses_node_bounds_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions.push(spur_core::config::PartitionConfig {
            name: "small".into(),
            default: false,
            state: "UP".into(),
            nodes: "ALL".into(),
            selector: Default::default(),
            max_time: None,
            default_time: None,
            max_nodes: Some(1),
            min_nodes: 1,
            allow_accounts: Vec::new(),
            allow_groups: Vec::new(),
            deny_accounts: Vec::new(),
            deny_qos: Vec::new(),
            allow_qos: Vec::new(),
            priority_tier: 1,
            preempt_mode: String::new(),
            preempt_exempt_time: None,
        });
        config.hooks.job_submit_lua = Some(write_lua_script(
            &dir,
            "function slurm_job_submit(j, u)\n  j.partition = 'small'\n  return slurm.SUCCESS\nend",
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("lua-bounds-bypass");
        spec.num_nodes = 2;
        spec.num_tasks = 2;
        let err = cm.submit_job(spec).unwrap_err();
        assert!(
            err.to_string().contains("outside partition"),
            "expected a node-bounds rejection, got: {err:?}"
        );
    }

    // Shell runs first, then Lua, each on the evolving spec.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_shell_then_lua_chain() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit = Some(write_hook_script(&dir, r#"echo '{"priority": 100}'"#));
        config.hooks.job_submit_lua = Some(write_lua_script(
            &dir,
            "function slurm_job_submit(j, u)\n  j.comment = 'chained'\n  return slurm.SUCCESS\nend",
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let id = submit_and_wait(&cm, basic_spec("chain"));
        let job = cm.get_job(id).unwrap();
        assert_eq!(job.spec.priority, Some(100));
        assert_eq!(job.spec.comment.as_deref(), Some("chained"));
    }

    // A shell rejection short-circuits the chain: the Lua hook never runs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_shell_reject_skips_lua() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit = Some(write_hook_script(&dir, "echo 'shell says no' >&2\nexit 1"));
        // If this Lua ran it would mkdir a marker; assert the file never appears.
        let marker = dir.path().join("lua-ran");
        config.hooks.job_submit_lua = Some(write_lua_script(
            &dir,
            &format!(
                "function slurm_job_submit(j, u)\n  j.comment = '{}'\n  return slurm.SUCCESS\nend",
                marker.display()
            ),
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let err = cm.submit_job(basic_spec("shell-first")).unwrap_err();
        assert_eq!(err, SubmitError::invalid("shell says no"));
    }

    // When both hooks set the same field, Lua (second) wins on the evolving spec.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_lua_overrides_shell_field() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.hooks.job_submit = Some(write_hook_script(
            &dir,
            r#"echo '{"comment": "from-shell"}'"#,
        ));
        config.hooks.job_submit_lua = Some(write_lua_script(
            &dir,
            "function slurm_job_submit(j, u)\n  j.comment = 'from-lua'\n  return slurm.SUCCESS\nend",
        ));
        let cm = test_cluster_with_config(&dir, config).await;

        let id = submit_and_wait(&cm, basic_spec("override"));
        assert_eq!(
            cm.get_job(id).unwrap().spec.comment.as_deref(),
            Some("from-lua")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cold_start_respects_first_job_id() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.controller.first_job_id = 500;
        let cm = test_cluster_with_config(&dir, config).await;

        assert_eq!(cm.next_job_id.load(Ordering::Relaxed), 500);

        let job_id = submit_and_wait(&cm, basic_spec("first-job"));
        assert_eq!(job_id, 500);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_state_change() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("j")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Running);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_start_allocates_resources() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "node1", 8, 16000);
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("j")),
        });

        let resources = scalar_alloc(4, 8000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["node1".into()],
            resources: resources.clone(),
            per_node_alloc: per_node_for(&["node1"], resources),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        let job = cm.get_job(1).unwrap();
        assert!(job.start_time.is_some());
        assert_eq!(job.allocated_nodes, vec!["node1"]);

        let node = cm.get_node("node1").unwrap();
        assert_eq!(node.alloc_resources.cpus, 4);
        assert_eq!(node.alloc_resources.memory_mb, 8000);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_complete_deallocates_resources() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "node1", 8, 16000);
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("j")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(4, 8000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["node1".into()],
            resources: alloc.clone(),
            per_node_alloc: per_node_for(&["node1"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        cm.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: 0,
            state: JobState::Completed,
        });

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Completed);
        assert_eq!(job.exit_code, Some(0));
        assert!(job.end_time.is_some());

        let node = cm.get_node("node1").unwrap();
        assert_eq!(node.alloc_resources.cpus, 0);
        assert_eq!(node.alloc_resources.memory_mb, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evict_terminal_jobs_drops_only_aged_terminal_jobs() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("done")),
        });
        cm.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: 0,
            state: JobState::Cancelled,
        });
        assert!(
            cm.get_job(1).unwrap().state.is_terminal(),
            "job 1 is terminal"
        );
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 2,
            spec: Box::new(basic_spec("running")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            2,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 3,
            spec: Box::new(basic_spec("pending")),
        });
        // A job stranded in Preempted (a rare requeue-strand): finalized with an
        // end_time, so it must be reapable even though it isn't is_terminal().
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 4,
            spec: Box::new(basic_spec("preempted")),
        });
        {
            let mut jobs = cm.jobs.write();
            let j = jobs.get_mut(&4).unwrap();
            j.state = JobState::Preempted;
            j.end_time = Some(chrono::Utc::now());
        }
        assert!(!cm.get_job(4).unwrap().state.is_terminal());
        assert!(cm.get_job(4).unwrap().state.is_finalized());

        // A step on each of the terminal (1) and live (2) jobs: eviction must
        // drop the evicted job's step and keep the live one's.
        let step = |job_id: JobId| JobStep {
            job_id,
            step_id: 0,
            name: "s".into(),
            state: StepState::Running,
            num_tasks: 1,
            cpus_per_task: 1,
            resources: scalar_alloc(1, 0),
            nodes: vec!["node1".into()],
            distribution: spur_core::step::TaskDistribution::Block,
            start_time: None,
            end_time: None,
            exit_code: None,
        };
        cm.steps.write().insert((1, 0), step(1));
        cm.steps.write().insert((2, 0), step(2));

        // Apply removes only ids still finalized; live ids 2 and 3 are no-ops
        // even when named (a job requeued between propose and apply).
        cm.apply_operation(&WalOperation::EvictTerminalJobs {
            job_ids: vec![1, 2, 3, 4],
        });
        assert!(
            cm.get_job(1).is_none(),
            "named terminal job must be evicted"
        );
        assert!(cm.get_job(2).is_some(), "running job must be spared");
        assert!(cm.get_job(3).is_some(), "pending job must be spared");
        assert!(
            cm.get_job(4).is_none(),
            "named finalized (Preempted) job must be evicted"
        );
        assert!(
            cm.steps.read().get(&(1, 0)).is_none(),
            "evicted job's step must be removed"
        );
        assert!(
            cm.steps.read().get(&(2, 0)).is_some(),
            "live job's step must be kept"
        );
    }

    // Apply-time state, not per-replica end_time, decides eviction: a replica
    // where the id requeued before apply spares it; skew never changes the set.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evict_terminal_jobs_is_replica_deterministic() {
        let dir_a = TempDir::new().unwrap();
        let dir_b = TempDir::new().unwrap();
        let cm_a = test_cluster(&dir_a).await;
        let cm_b = test_cluster(&dir_b).await;

        for cm in [&cm_a, &cm_b] {
            cm.apply_operation(&WalOperation::JobSubmit {
                job_id: 1,
                spec: Box::new(basic_spec("done")),
            });
            cm.apply_operation(&WalOperation::JobComplete {
                job_id: 1,
                exit_code: 0,
                state: JobState::Cancelled,
            });
        }
        // Divergent local end_times: skew must not change the outcome, since the
        // apply guard reads state, not end_time.
        cm_a.jobs.write().get_mut(&1).unwrap().end_time =
            Some(chrono::Utc::now() - chrono::Duration::hours(2));
        cm_b.jobs.write().get_mut(&1).unwrap().end_time =
            Some(chrono::Utc::now() + chrono::Duration::hours(2));

        let entry = WalOperation::EvictTerminalJobs { job_ids: vec![1] };
        cm_a.apply_operation(&entry);
        cm_b.apply_operation(&entry);
        assert!(cm_a.get_job(1).is_none(), "finalized id is evicted");
        assert!(
            cm_b.get_job(1).is_none(),
            "end_time skew must not change the evicted set",
        );

        // A replica where the id left the finalized set before apply spares it:
        // the guard is what makes eviction converge with real state, not the id.
        let dir_c = TempDir::new().unwrap();
        let cm_c = test_cluster(&dir_c).await;
        cm_c.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("requeued")),
        });
        cm_c.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        cm_c.apply_operation(&entry);
        assert!(
            cm_c.get_job(1).is_some(),
            "an id no longer finalized at apply is spared",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evict_expired_terminal_jobs_honors_retention_window() {
        // A large retention window spares a fresh terminal job: guards against a
        // producer that proposes everything regardless of age.
        let dir2 = TempDir::new().unwrap();
        let mut cfg2 = test_config();
        cfg2.controller.terminal_job_retention_secs = 86_400;
        let cm2 = test_cluster_with_config(&dir2, cfg2).await;
        cm2.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("fresh")),
        });
        cm2.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: 0,
            state: JobState::Cancelled,
        });
        cm2.evict_expired_terminal_jobs();
        assert!(
            cm2.get_job(1).is_some(),
            "a fresh terminal job within the retention window is spared"
        );

        // An aged job (end_time well past the window) is evicted.
        cm2.jobs.write().get_mut(&1).unwrap().end_time =
            Some(chrono::Utc::now() - chrono::Duration::days(2));
        cm2.evict_expired_terminal_jobs();
        assert!(
            cm2.get_job(1).is_none(),
            "a terminal job older than the retention window is evicted"
        );
    }

    // retention below the reconcile interval is floored so a job survives at
    // least one reconcile pass, else its DB row strands as RUNNING in sacct.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evict_floors_retention_to_reconcile_interval() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.controller.terminal_job_retention_secs = 0;
        let cm = test_cluster_with_config(&dir, cfg).await;

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("done")),
        });
        cm.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: 0,
            state: JobState::Cancelled,
        });

        // Just completed: within the floored window, so retention 0 does not
        // evict on the next tick.
        cm.evict_expired_terminal_jobs();
        assert!(
            cm.get_job(1).is_some(),
            "retention 0 must be floored above the reconcile interval, not evict immediately"
        );

        // Older than the floor: now evictable.
        cm.jobs.write().get_mut(&1).unwrap().end_time = Some(
            chrono::Utc::now()
                - chrono::Duration::seconds(crate::accounting::RECONCILE_INTERVAL_SECS as i64 + 1),
        );
        cm.evict_expired_terminal_jobs();
        assert!(
            cm.get_job(1).is_none(),
            "a job past the floored window is evicted"
        );
    }

    // A target referenced by a pending job's dependency must not be evicted:
    // dropping it would cancel the afterok dependent before its trigger fires.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evict_spares_jobs_referenced_by_pending_dependency() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.controller.terminal_job_retention_secs = 0;
        let cm = test_cluster_with_config(&dir, cfg).await;

        // Target runs, completes, and ages out of the window.
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 100,
            spec: Box::new(basic_spec("target")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            100,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobComplete {
            job_id: 100,
            exit_code: 0,
            state: JobState::Completed,
        });
        cm.jobs.write().get_mut(&100).unwrap().end_time =
            Some(chrono::Utc::now() - chrono::Duration::days(1));

        // Pending child depends on it (begin-time hold, not a scheduling hold).
        let mut child = basic_spec("child");
        child.dependency = vec!["afterok:100".into()];
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 101,
            spec: Box::new(child),
        });

        cm.evict_expired_terminal_jobs();
        assert!(
            cm.get_job(100).is_some(),
            "a target referenced by a pending afterok dependent must be spared"
        );

        // Once the child is gone, the target is evictable.
        cm.apply_operation(&WalOperation::JobComplete {
            job_id: 101,
            exit_code: 0,
            state: JobState::Cancelled,
        });
        cm.jobs.write().get_mut(&101).unwrap().end_time =
            Some(chrono::Utc::now() - chrono::Duration::days(1));
        cm.evict_expired_terminal_jobs();
        assert!(
            cm.get_job(100).is_none(),
            "with no live dependent, the aged target is evicted"
        );
    }

    // Eviction removes the high-id tail; a snapshot+restore must not lower
    // next_job_id and reissue a used id (which would clobber the sacct row).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_restore_preserves_next_job_id_after_eviction() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // Allocate ids 1 and 2. id1 survives; the high-id tail (id2) completes
        // and is evicted, so rebuild-from-survivors alone would yield id1+1.
        let id1 = cm.next_job_id.fetch_add(1, Ordering::SeqCst);
        let id2 = cm.next_job_id.fetch_add(1, Ordering::SeqCst);
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: id1,
            spec: Box::new(basic_spec("survivor")),
        });
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: id2,
            spec: Box::new(basic_spec("high")),
        });
        cm.apply_operation(&WalOperation::JobComplete {
            job_id: id2,
            exit_code: 0,
            state: JobState::Cancelled,
        });
        let next_before = cm.next_job_id.load(Ordering::Relaxed);
        cm.apply_operation(&WalOperation::EvictTerminalJobs { job_ids: vec![id2] });
        assert!(cm.get_job(id2).is_none(), "high-id job evicted");

        // Snapshot AFTER eviction: id2's JobSubmit is compacted away, so the map
        // no longer contains the high id.
        let snap = cm.snapshot_state().unwrap();

        // Restore into a fresh controller: surviving jobs alone would rebuild
        // next_job_id below next_before, reissuing id2.
        let dir2 = TempDir::new().unwrap();
        let cm2 = test_cluster(&dir2).await;
        cm2.restore_from_snapshot(&snap).unwrap();
        assert!(
            cm2.next_job_id.load(Ordering::Relaxed) >= next_before,
            "next_job_id must not regress below {next_before} after evicting id {id2} (was {}), id {id1} survives",
            cm2.next_job_id.load(Ordering::Relaxed)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_suspend_then_resume_accumulates_suspended_secs() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("s")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let t0 = chrono::Utc::now();
        cm.apply_operation(&WalOperation::JobSuspend {
            job_id: 1,
            at: t0,
            preempted_by: None,
            preempt_qos: None,
        });
        assert_eq!(cm.get_job(1).unwrap().state, JobState::Suspended);
        cm.apply_operation(&WalOperation::JobResume {
            job_id: 1,
            at: t0 + chrono::Duration::seconds(25),
        });
        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Running);
        assert_eq!(job.suspended_secs, 25);
        assert!(job.suspended_at.is_none());
    }

    // ── suspend_job / resume_job method guards ───────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn suspend_job_rejects_pending() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let id = submit_and_wait(&cm, basic_spec("p"));
        // Job is Pending (never started).
        let err = cm.suspend_job(id, "u").unwrap_err();
        assert!(
            err.to_string().contains("not running"),
            "unexpected error: {err}"
        );
        assert_eq!(cm.get_job(id).unwrap().state, JobState::Pending);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn resume_job_rejects_pending() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let id = submit_and_wait(&cm, basic_spec("p"));
        let err = cm.resume_job(id, "u").unwrap_err();
        assert!(
            err.to_string().contains("not suspended"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn resume_job_rejects_running() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let id = submit_and_wait(&cm, basic_spec("r"));
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);
        // Resuming a running (not suspended) job is rejected.
        assert!(cm.resume_job(id, "u").is_err());
        assert_eq!(cm.get_job(id).unwrap().state, JobState::Running);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn suspend_resume_unknown_job_errors() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        assert!(cm
            .suspend_job(9999, "u")
            .unwrap_err()
            .to_string()
            .contains("not found"));
        assert!(cm
            .resume_job(9999, "u")
            .unwrap_err()
            .to_string()
            .contains("not found"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn double_suspend_is_rejected() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let id = submit_and_wait(&cm, basic_spec("d"));
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);
        cm.suspend_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Suspended);
        // Second suspend on an already-suspended job is rejected (not Running).
        assert!(cm.suspend_job(id, "testuser").is_err());
        assert_eq!(cm.get_job(id).unwrap().state, JobState::Suspended);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn double_resume_is_rejected() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let id = submit_and_wait(&cm, basic_spec("d"));
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);
        cm.suspend_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Suspended);
        cm.resume_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Running);
        // Second resume on an already-running job is rejected.
        assert!(cm.resume_job(id, "testuser").is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn suspend_retains_node_allocation() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let id = submit_and_wait(&cm, basic_spec("a"));
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);
        assert_eq!(cm.get_node("n1").unwrap().alloc_resources.cpus, 2);

        cm.suspend_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Suspended);
        // Allocation is retained while suspended (plain scontrol suspend parity).
        let job = cm.get_job(id).unwrap();
        assert_eq!(job.allocated_nodes, vec!["n1".to_string()]);
        assert_eq!(
            cm.get_node("n1").unwrap().alloc_resources.cpus,
            2,
            "node resources must stay allocated while job is suspended"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn two_suspend_cycles_accumulate_seconds() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("acc")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let t0 = chrono::Utc::now();
        // Cycle 1: 10s suspended.
        cm.apply_operation(&WalOperation::JobSuspend {
            job_id: 1,
            at: t0,
            preempted_by: None,
            preempt_qos: None,
        });
        cm.apply_operation(&WalOperation::JobResume {
            job_id: 1,
            at: t0 + chrono::Duration::seconds(10),
        });
        // Cycle 2: 15s suspended.
        let t1 = t0 + chrono::Duration::seconds(40);
        cm.apply_operation(&WalOperation::JobSuspend {
            job_id: 1,
            at: t1,
            preempted_by: None,
            preempt_qos: None,
        });
        cm.apply_operation(&WalOperation::JobResume {
            job_id: 1,
            at: t1 + chrono::Duration::seconds(15),
        });
        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Running);
        assert_eq!(job.suspended_secs, 25, "10 + 15 accumulated");
        assert!(job.suspended_at.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_while_suspended_finalizes_suspended_at() {
        // Copilot review: a Suspended -> terminal transition must clear
        // suspended_at (so it never lingers on a terminal job and
        // `suspended_at.is_some()` keeps meaning "currently suspended") and fold
        // the final suspended interval into suspended_secs.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("cancel-susp")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        // Suspended 30s ago, then cancelled now (JobComplete stamps Utc::now()).
        let since = chrono::Utc::now() - chrono::Duration::seconds(30);
        cm.apply_operation(&WalOperation::JobSuspend {
            job_id: 1,
            at: since,
            preempted_by: None,
            preempt_qos: None,
        });
        cm.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: 0,
            state: JobState::Cancelled,
        });
        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Cancelled);
        assert!(
            job.suspended_at.is_none(),
            "suspended_at must be cleared on a Suspended -> terminal transition"
        );
        assert!(
            job.suspended_secs >= 30,
            "final suspended interval folded into suspended_secs (got {})",
            job.suspended_secs
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn suspended_job_excluded_from_timelimit_scan() {
        // The time-limit enforcer scans only [Running, Completing] jobs, so a
        // suspended job is never warned/killed while frozen. Assert the exact
        // query the enforcer uses does not return a suspended job.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let id = submit_and_wait(&cm, basic_spec("t"));
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);
        cm.suspend_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Suspended);

        let scanned = cm.get_jobs(&JobFilter {
            states: &[JobState::Running, JobState::Completing],
            ..Default::default()
        });
        assert!(
            !scanned.iter().any(|j| j.job_id == id),
            "suspended job must not appear in the enforcer's Running/Completing scan"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_node_register() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.apply_operation(&WalOperation::NodeRegister {
            name: "gpu-node".into(),
            hostname: String::new(),
            resources: ResourceSet {
                cpus: 64,
                memory_mb: 256000,
                ..Default::default()
            },
            address: "10.0.0.1".into(),
            port: 6818,
            wg_pubkey: String::new(),
            version: "1.0".into(),
            labels: HashMap::new(),
            source: NodeSource::default(),
        });

        let node = cm.get_node("gpu-node").unwrap();
        assert_eq!(node.total_resources.cpus, 64);
        assert_eq!(node.state, NodeState::Idle);
        assert_eq!(node.address, Some("10.0.0.1".into()));
        assert_eq!(node.hostname, "gpu-node");
        // Dynamically registered nodes get the default partition
        assert!(
            !node.partitions.is_empty(),
            "node should be assigned to default partition"
        );
        assert_eq!(node.partitions[0], "default");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_node_state_change() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);
        cm.apply_operation(&WalOperation::NodeStateChange {
            name: "n1".into(),
            old_state: NodeState::Idle,
            new_state: NodeState::Drain,
            reason: Some("maintenance".into()),
            admin_locked: true,
            reason_uid: None,
            reason_time: None,
        });

        let node = cm.get_node("n1").unwrap();
        assert_eq!(node.state, NodeState::Drain);
        assert_eq!(node.state_reason, Some("maintenance".into()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_priority_change() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("j")),
        });
        cm.apply_operation(&WalOperation::JobPriorityChange {
            job_id: 1,
            old_priority: 1000,
            new_priority: 5000,
            pending_reason: None,
            pending_reason_desc: None,
            reset_requeue_count: false,
            clear_reservation: false,
        });

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.priority, 5000);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_assigns_id_and_applies() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let id = submit_and_wait(&cm, basic_spec("my-job"));
        assert!(id >= 1);

        let job = cm.get_job(id).unwrap();
        assert_eq!(job.spec.name, "my-job");
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.spec.partition, Some("default".into()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_reduces_nodes_to_task_count() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let mut spec = basic_spec("fewer-tasks");
        spec.num_nodes = 4;
        spec.num_tasks = 1;
        let id = submit_and_wait(&cm, spec);

        // The persisted spec (what all reporting reads) reflects the node count
        // actually allocatable, not the over-request.
        assert_eq!(cm.get_job(id).unwrap().spec.num_nodes, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_keeps_nodes_when_tasks_match() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let mut spec = basic_spec("full-fit");
        spec.num_nodes = 4;
        spec.num_tasks = 4;
        let id = submit_and_wait(&cm, spec);

        assert_eq!(cm.get_job(id).unwrap().spec.num_nodes, 4);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_keeps_nodes_with_explicit_tasks_per_node() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let mut spec = basic_spec("per-node-layout");
        spec.num_nodes = 4;
        spec.num_tasks = 1;
        spec.tasks_per_node = Some(2);
        let id = submit_and_wait(&cm, spec);

        // An explicit per-node layout pins the node count regardless of the
        // task total.
        assert_eq!(cm.get_job(id).unwrap().spec.num_nodes, 4);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_accepts_gpu_after_node_reduction() {
        // C3: -N4 -n1 --gpus=2 is valid once the node count is reduced to one;
        // the pre-normalization guard used to reject it.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let mut spec = basic_spec("gpu-reduced");
        spec.num_nodes = 4;
        spec.num_tasks = 1;
        spec.gpus = Some(spur_core::gpu_request::GpuRequest::new(2, None));
        let outcome = cm.submit_job(spec).unwrap();
        wait_for("job applied", || cm.get_job(outcome.job_id).is_some());
        assert_eq!(cm.get_job(outcome.job_id).unwrap().spec.num_nodes, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_warns_when_node_count_reduced() {
        // I2: a shrunk request returns a user-facing warning; an exact-fit
        // request returns none.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let mut shrunk = basic_spec("shrunk");
        shrunk.num_nodes = 4;
        shrunk.num_tasks = 1;
        let outcome = cm.submit_job(shrunk).unwrap();
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.contains("requested 4 nodes but only 1")),
            "expected a node-reduction warning, got {:?}",
            outcome.warnings
        );

        let mut exact = basic_spec("exact");
        exact.num_nodes = 4;
        exact.num_tasks = 4;
        let outcome = cm.submit_job(exact).unwrap();
        assert!(outcome.warnings.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_below_partition_min_nodes() {
        // I1: -N4 -n1 on a MinNodes=4 partition is rejected at submit, not left
        // pending forever.
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].min_nodes = 4;
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("undersize");
        spec.partition = Some("default".into());
        spec.num_nodes = 4;
        spec.num_tasks = 1; // reduces to 1 node, below min 4
        let err = cm.submit_job(spec).unwrap_err();
        assert!(matches!(err, SubmitError::InvalidArgument(_)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_above_partition_max_nodes() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].max_nodes = Some(1);
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("oversize");
        spec.partition = Some("default".into());
        spec.num_nodes = 2;
        spec.num_tasks = 2;
        assert!(cm.submit_job(spec).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_accepts_within_partition_node_bounds() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].min_nodes = 1;
        config.partitions[0].max_nodes = Some(4);
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("fits");
        spec.partition = Some("default".into());
        spec.num_nodes = 2;
        spec.num_tasks = 2;
        assert!(cm.submit_job(spec).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_pmix_accepts_single_node_after_reduction() {
        // I5: -N4 -n1 --mpi=pmix reduces to one node and is accepted.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let mut spec = basic_spec("pmix-one");
        spec.num_nodes = 4;
        spec.num_tasks = 1;
        spec.mpi = Some(spur_core::mpi::MPI_PMIX.into());
        assert!(cm.submit_job(spec).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_pmix_accepts_multi_node_on_native_hosts() {
        // Multi-node --mpi=pmix is allowed on native hosts (K8s agents reject at dispatch).
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let mut spec = basic_spec("pmix-many");
        spec.num_nodes = 4;
        spec.num_tasks = 4;
        spec.mpi = Some(spur_core::mpi::MPI_PMIX.into());
        assert!(cm.submit_job(spec).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_keeps_full_node_count_for_total_gpus() {
        // C2: -N4 --gpus=8 (ntasks defaulted to nodes) keeps all four nodes so
        // the eight GPUs spread across them instead of collapsing to one node.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let mut spec = basic_spec("gpu-spread");
        spec.num_nodes = 4;
        spec.num_tasks = 4; // one task per node (adapter default for absent -n)
        spec.gpus = Some(spur_core::gpu_request::GpuRequest::new(8, None));
        let outcome = cm.submit_job(spec).unwrap();
        wait_for("job applied", || cm.get_job(outcome.job_id).is_some());
        assert_eq!(cm.get_job(outcome.job_id).unwrap().spec.num_nodes, 4);
        assert!(outcome.warnings.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_multiple_jobs_increments_ids() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let id1 = submit_and_wait(&cm, basic_spec("a"));
        let id2 = submit_and_wait(&cm, basic_spec("b"));
        let id3 = submit_and_wait(&cm, basic_spec("c"));

        assert!(id2 > id1);
        assert!(id3 > id2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn start_and_complete_job_lifecycle() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "worker1", 8, 16000);
        let job_id = submit_and_wait(&cm, basic_spec("lifecycle"));

        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Running);
        assert!(job.start_time.is_some());

        let node = cm.get_node("worker1").unwrap();
        assert_eq!(node.alloc_resources.cpus, 2);

        cm.complete_job(job_id, 0, JobState::Completed).unwrap();
        settle(&cm, job_id, JobState::Completed);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Completed);
        assert_eq!(job.exit_code, Some(0));

        let node = cm.get_node("worker1").unwrap();
        assert_eq!(node.alloc_resources.cpus, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn planned_reservation_round_trips_and_defaults_to_none() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        assert_eq!(cm.planned_reservation("node001"), None);

        let now = Utc::now();
        let mut planned = HashMap::new();
        planned.insert("node001".to_string(), (7u32, now));
        cm.set_planned_reservations(planned);

        assert_eq!(cm.planned_reservation("node001"), Some((7, now)));
        assert_eq!(cm.planned_reservation("node002"), None);

        // A later cycle with nothing planned clears the stale entry.
        cm.set_planned_reservations(HashMap::new());
        assert_eq!(cm.planned_reservation("node001"), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sched_stats_track_submit_start_complete() {
        use std::sync::Arc;

        use crate::sched_stats::SchedStatsCollector;

        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let stats = Arc::new(SchedStatsCollector::new("backfill"));
        cm.set_sched_stats(stats.clone());

        register_node(&cm, "worker1", 8, 16000);
        let job_id = submit_and_wait(&cm, basic_spec("stats-job"));
        assert_eq!(stats.snapshot().jobs_submitted, 1);

        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        cm.record_sched_cycle(0, 0, 1, false);
        assert_eq!(stats.snapshot().jobs_started, 1);

        cm.complete_job(job_id, 0, JobState::Completed).unwrap();
        settle(&cm, job_id, JobState::Completed);
        assert_eq!(stats.snapshot().jobs_finalized, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sched_stats_track_preempted_jobs() {
        use std::sync::Arc;

        use crate::sched_stats::SchedStatsCollector;

        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let stats = Arc::new(SchedStatsCollector::new("backfill"));
        cm.set_sched_stats(stats.clone());

        register_node(&cm, "worker1", 8, 16000);

        let job1 = run_job_on(&cm, "victim-1", "worker1");
        cm.preempt_job_with_provenance(job1, PreemptMode::Requeue, Some(99), None)
            .unwrap();
        settle(&cm, job1, JobState::Pending);
        assert_eq!(
            stats.snapshot().jobs_preempted,
            1,
            "requeue preemption counted"
        );

        let job2 = run_job_on(&cm, "victim-2", "worker1");
        cm.preempt_job_with_provenance(job2, PreemptMode::Cancel, Some(99), None)
            .unwrap();
        settle(&cm, job2, JobState::Cancelled);
        assert_eq!(
            stats.snapshot().jobs_preempted,
            2,
            "cancel preemption counted"
        );

        assert_eq!(
            stats.snapshot().jobs_finalized,
            2,
            "both preemptions counted as finalized"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_node_complete_single_node() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "worker1", 8, 16000);
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("single-completing")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["worker1".into()],
            resources: alloc.clone(),
            per_node_alloc: per_node_for(&["worker1"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "worker1".into(),
            exit_code: 0,
            signal: 0,
        });

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Completed);
        assert_eq!(job.exit_code, Some(0));
        assert!(job.node_completions.is_empty());
        assert_eq!(cm.get_node("worker1").unwrap().alloc_resources.cpus, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_node_complete_oom_sets_out_of_memory() {
        // spurd reports an OOM kill as SIGKILL with the OOM sentinel bit OR'd in.
        // The job must finalize as OUT_OF_MEMORY / Reason=OutOfMemory, with the
        // sentinel stripped so the stored signal is the real SIGKILL (9).
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("oom")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["worker1".into()],
            resources: alloc.clone(),
            per_node_alloc: per_node_for(&["worker1"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "worker1".into(),
            exit_code: 0,
            signal: spur_core::job::OOM_SIGNAL_FLAG | 9,
        });

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::OutOfMemory);
        assert_eq!(job.pending_reason, PendingReason::OutOfMemory);
        assert_eq!(job.exit_signal, 9, "OOM sentinel must be stripped");
        assert_eq!(cm.get_node("worker1").unwrap().alloc_resources.cpus, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_node_complete_multi_node() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        for name in ["n1", "n2", "n3"] {
            register_node(&cm, name, 8, 16000);
        }

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("multi-completing")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into(), "n2".into(), "n3".into()],
            resources: scalar_alloc(6, 12000),
            per_node_alloc: per_node_for(&["n1", "n2", "n3"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n1".into(),
            exit_code: 0,
            signal: 0,
        });
        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Completing);
        assert_eq!(job.node_completions.len(), 1);
        assert_eq!(cm.get_node("n1").unwrap().alloc_resources.cpus, 0);
        assert!(cm.get_node("n2").unwrap().alloc_resources.cpus > 0);

        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n2".into(),
            exit_code: 0,
            signal: 0,
        });
        assert_eq!(cm.get_job(1).unwrap().state, JobState::Completing);

        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n3".into(),
            exit_code: 42,
            signal: 0,
        });

        let job = cm.get_job(1).unwrap();
        // ExitCode follows the primary (batch) node n1 = allocated_nodes[0],
        // which exited 0 — so the job state/exit_code reflect a clean primary.
        assert_eq!(job.state, JobState::Completed);
        assert_eq!(job.exit_code, Some(0));
        // DerivedExitCode is the max over srun *steps* (Slurm parity), not node
        // completions. This job ran no srun steps, so it is 0 — the non-primary
        // node's exit 42 does not surface here.
        assert_eq!(job.derived_exit_code, 0);
        for name in ["n1", "n2", "n3"] {
            assert_eq!(cm.get_node(name).unwrap().alloc_resources.cpus, 0);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn step_complete_accumulates_derived_exit_code_running_max() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("steps")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into()],
            resources: scalar_alloc(4, 8000),
            per_node_alloc: per_node_for(&["n1"], scalar_alloc(4, 8000)),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        // Three srun steps exit 7, 3, 2 (in that order). DerivedExitCode tracks
        // the running max live; ExitCode is unaffected (it is the batch exit).
        cm.apply_operation(&WalOperation::JobStepComplete {
            job_id: 1,
            step_id: 0,
            exit_code: 7,
        });
        assert_eq!(cm.get_job(1).unwrap().derived_exit_code, 7);
        cm.apply_operation(&WalOperation::JobStepComplete {
            job_id: 1,
            step_id: 1,
            exit_code: 3,
        });
        // 3 < 7, running max stays 7.
        assert_eq!(cm.get_job(1).unwrap().derived_exit_code, 7);
        cm.apply_operation(&WalOperation::JobStepComplete {
            job_id: 1,
            step_id: 2,
            exit_code: 2,
        });
        assert_eq!(cm.get_job(1).unwrap().derived_exit_code, 7);

        // Batch script exits 2 -> ExitCode=2:0, DerivedExitCode preserved at 7.
        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n1".into(),
            exit_code: 2,
            signal: 0,
        });
        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Failed);
        assert_eq!(job.exit_code, Some(2));
        assert_eq!(job.derived_exit_code, 7);
    }

    /// A PTY client reports client-side, so its report can land after a sweep
    /// has already closed the step out. The sweep's verdict must stand.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn step_complete_is_ignored_once_the_job_is_terminal() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("late-step")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into()],
            resources: scalar_alloc(4, 8000),
            per_node_alloc: per_node_for(&["n1"], scalar_alloc(4, 8000)),
            srun_step_dispatch: false,
            run_attempt: 0,
        });
        cm.apply_operation(&WalOperation::JobStepCreate {
            step: Box::new(spur_core::step::JobStep {
                job_id: 1,
                step_id: 0,
                name: "pty".into(),
                state: StepState::Running,
                num_tasks: 1,
                cpus_per_task: 1,
                resources: spur_core::resource::ResourceAllocations::default(),
                nodes: vec!["n1".into()],
                distribution: spur_core::step::TaskDistribution::Block,
                start_time: Some(Utc::now()),
                end_time: None,
                exit_code: None,
            }),
        });
        cm.apply_operation(&WalOperation::JobStepComplete {
            job_id: 1,
            step_id: 0,
            exit_code: 3,
        });
        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n1".into(),
            exit_code: 0,
            signal: 0,
        });

        let swept = cm
            .get_step(1, 0)
            .expect("step survives job completion until eviction");

        cm.apply_operation(&WalOperation::JobStepComplete {
            job_id: 1,
            step_id: 0,
            exit_code: 9,
        });

        let after = cm.get_step(1, 0).expect("step still present");
        assert_eq!(after.exit_code, swept.exit_code, "verdict must not reopen");
        assert_eq!(after.state, swept.state);
        assert_eq!(
            cm.get_job(1).unwrap().derived_exit_code,
            3,
            "a late report must not raise DerivedExitCode after accounting saw it"
        );
    }

    /// The requeue's own accounting row is written after that apply returns, so
    /// the previous run's DerivedExitCode must survive it and clear at next start.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn derived_exit_code_survives_requeue_and_clears_at_next_start() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        let start = || WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into()],
            resources: scalar_alloc(4, 8000),
            per_node_alloc: per_node_for(&["n1"], scalar_alloc(4, 8000)),
            srun_step_dispatch: false,
            run_attempt: 0,
        };

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("requeued")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&start());
        cm.apply_operation(&WalOperation::JobStepComplete {
            job_id: 1,
            step_id: 0,
            exit_code: 5,
        });
        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n1".into(),
            exit_code: 0,
            signal: 9,
        });
        let job = cm.get_job(1).unwrap();
        assert_eq!((job.derived_exit_code, job.exit_signal), (5, 9));

        cm.apply_operation(&WalOperation::JobUserRequeue {
            job_id: 1,
            hold: false,
            begin_time: None,
        });
        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(
            (job.derived_exit_code, job.exit_signal),
            (5, 9),
            "accounting for the finished run is written after this apply"
        );

        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&start());
        let job = cm.get_job(1).unwrap();
        assert_eq!(
            (job.derived_exit_code, job.exit_signal),
            (0, 0),
            "the new run must not inherit the previous run's exit fields"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn step_complete_batch_step_excluded_from_derived() {
        // The reserved batch step carries the job's own exit, not a step result,
        // so it must NOT contribute to DerivedExitCode.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("batch-only")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));

        cm.apply_operation(&WalOperation::JobStepComplete {
            job_id: 1,
            step_id: STEP_BATCH,
            exit_code: 9,
        });
        // Reserved step id -> derived untouched.
        assert_eq!(cm.get_job(1).unwrap().derived_exit_code, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_node_complete_returns_finalized_once() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        for name in ["n1", "n2"] {
            register_node(&cm, name, 8, 16000);
        }

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("finalize-response")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into(), "n2".into()],
            resources: scalar_alloc(4, 8000),
            per_node_alloc: per_node_for(&["n1", "n2"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        let r1 = cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n1".into(),
            exit_code: 0,
            signal: 0,
        });
        assert!(r1.jobs_finalized.is_empty());
        assert_eq!(cm.get_job(1).unwrap().state, JobState::Completing);

        let r2 = cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n2".into(),
            exit_code: 0,
            signal: 0,
        });
        let f = r2
            .jobs_finalized
            .first()
            .expect("last node should finalize");
        assert_eq!(f.job_id, 1);
        assert_eq!(f.state, JobState::Completed);
        assert_eq!(f.exit_code, 0);
        assert_eq!(cm.get_job(1).unwrap().state, JobState::Completed);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_complete_returns_finalized() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "worker1", 8, 16000);
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("job-complete-response")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["worker1".into()],
            resources: alloc.clone(),
            per_node_alloc: per_node_for(&["worker1"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        let resp = cm.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: 0,
            state: JobState::Completed,
        });
        let f = resp
            .jobs_finalized
            .first()
            .expect("JobComplete should finalize");
        assert_eq!(f.job_id, 1);
        assert_eq!(f.state, JobState::Completed);
        assert_eq!(f.exit_code, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_complete_noop_when_already_terminal() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "worker1", 8, 16000);
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("double-complete")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["worker1".into()],
            resources: alloc.clone(),
            per_node_alloc: per_node_for(&["worker1"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        let first = cm.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: 0,
            state: JobState::Completed,
        });
        assert!(
            !first.jobs_finalized.is_empty(),
            "first JobComplete should finalize"
        );
        let node = cm.get_node("worker1").unwrap();
        assert_eq!(node.alloc_resources.cpus, 0);
        assert_eq!(node.alloc_resources.memory_mb, 0);

        let second = cm.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: -1,
            state: JobState::Cancelled,
        });
        assert!(second.jobs_finalized.is_empty());

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Completed);
        assert_eq!(job.exit_code, Some(0));

        let node = cm.get_node("worker1").unwrap();
        assert_eq!(node.alloc_resources.cpus, 0);
        assert_eq!(node.alloc_resources.memory_mb, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_complete_penultimate_returns_completing() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        for name in ["n1", "n2", "n3"] {
            register_node(&cm, name, 8, 16000);
        }

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("penultimate")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into(), "n2".into(), "n3".into()],
            resources: scalar_alloc(6, 12000),
            per_node_alloc: per_node_for(&["n1", "n2", "n3"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });
        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n1".into(),
            exit_code: 0,
            signal: 0,
        });

        let result = cm.node_complete(1, "n2", 0, 0, 0).unwrap();
        assert_eq!(result, NodeCompleteResult::Completing);
        assert_eq!(cm.get_job(1).unwrap().state, JobState::Completing);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_complete_sets_signal_reason_and_derived() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 8, 16000);

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("signal-job")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into()],
            resources: scalar_alloc(6, 12000),
            per_node_alloc: per_node_for(&["n1"], scalar_alloc(6, 12000)),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        cm.node_complete(1, "n1", 0, 9, 0).unwrap();
        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Failed);
        assert_eq!(job.exit_code, Some(0));
        assert_eq!(job.exit_signal, 9);
        assert_eq!(job.derived_exit_code, 0);
        assert_eq!(job.pending_reason, PendingReason::RaisedSignal);
    }

    // A job that exits promptly on the watchdog's SIGTERM used to report FAILED:
    // its completion reached the controller well before the grace period was up,
    // and nothing durable recorded why it had been signalled.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_complete_after_a_time_limit_signal_reports_timeout() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "time-limit-job", "worker1");
        cm.signal_time_limit(job_id, Utc::now()).unwrap();
        wait_for("time limit expiry recorded", || {
            cm.get_job(job_id)
                .is_some_and(|j| j.time_limit_signaled_at.is_some())
        });

        // What spurd reports for a script that dies on SIGTERM.
        cm.node_complete(job_id, "worker1", 0, 15, 0).unwrap();

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Timeout);
        assert_eq!(job.pending_reason, PendingReason::TimeLimit);
        // The terminating signal is still reported, so ExitCode stays 0:15.
        assert_eq!(job.exit_code, Some(0));
        assert_eq!(job.exit_signal, 15);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_time_limit_signal_after_the_run_ended_is_a_noop() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        // The watchdog can lose the race: the job finished on its own just as
        // the deadline passed, so its verdict is already final.
        let job_id = run_job_on(&cm, "raced-job", "worker1");
        cm.node_complete(job_id, "worker1", 0, 0, 0).unwrap();
        settle(&cm, job_id, JobState::Completed);

        cm.signal_time_limit(job_id, Utc::now()).unwrap();

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Completed);
        assert_eq!(job.pending_reason, PendingReason::None);
        assert!(job.time_limit_signaled_at.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn force_kill_after_the_grace_period_reports_the_time_limit_reason() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        // A job that outlives the grace period is finalized by the watchdog
        // itself rather than by an agent report.
        let job_id = run_job_on(&cm, "grace-expired", "worker1");
        cm.signal_time_limit(job_id, Utc::now()).unwrap();
        cm.complete_job(job_id, -1, JobState::Timeout).unwrap();
        settle(&cm, job_id, JobState::Timeout);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.pending_reason, PendingReason::TimeLimit);
        assert_eq!(job.exit_code, Some(-1));
    }

    // The completing-timeout force-finish and srun completion paths finalize
    // through complete_job with a state derived from exit status alone, never
    // consulting the marker. A run the watchdog already signalled must still
    // report TIMEOUT through those paths, or a well-behaved job skips requeue.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn complete_job_promotes_a_signaled_run_to_timeout() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let record_marker = |job_id: JobId| {
            cm.signal_time_limit(job_id, Utc::now()).unwrap();
            wait_for("time limit expiry recorded", || {
                cm.get_job(job_id)
                    .is_some_and(|j| j.time_limit_signaled_at.is_some())
            });
        };

        // force_finish_completing_job passes Failed (e.g. a node never reported)
        // and its exit_code must survive the promotion.
        let failed = run_job_on(&cm, "completing-timeout", "worker1");
        record_marker(failed);
        cm.complete_job(failed, 1, JobState::Failed).unwrap();
        settle(&cm, failed, JobState::Timeout);
        let job = cm.get_job(failed).unwrap();
        assert_eq!(job.pending_reason, PendingReason::TimeLimit);
        assert_eq!(job.exit_code, Some(1));

        // finish_srun_job passes Completed for a handler that trapped SIGTERM
        // and exited 0; the run still expired, matching Job::completion_verdict.
        let clean = run_job_on(&cm, "srun-timeout", "worker1");
        record_marker(clean);
        cm.complete_job(clean, 0, JobState::Completed).unwrap();
        settle(&cm, clean, JobState::Timeout);
        assert_eq!(
            cm.get_job(clean).unwrap().pending_reason,
            PendingReason::TimeLimit
        );

        // The promotion is narrow: a cancel on a signalled run stays Cancelled,
        // since the user's intent is not a time-limit expiry.
        let cancelled = run_job_on(&cm, "cancel-wins", "worker1");
        record_marker(cancelled);
        cm.complete_job(cancelled, -1, JobState::Cancelled).unwrap();
        settle(&cm, cancelled, JobState::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn requeue_after_a_time_limit_kill_clears_the_marker() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let mut spec = basic_spec("requeue-after-timeout");
        spec.requeue = true;
        let job_id = submit_and_wait(&cm, spec);
        let alloc = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            alloc.clone(),
            per_node_for(&["worker1"], alloc),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        cm.signal_time_limit(job_id, Utc::now()).unwrap();
        cm.node_complete(job_id, "worker1", 0, 15, 0).unwrap();

        // Attributing the kill to the time limit is what routes a well-behaved
        // job into the requeue path at all.
        settle(&cm, job_id, JobState::Pending);
        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.requeue_count, 1);
        // A marker left behind would make the next run report TIMEOUT the
        // moment it ended, whatever its outcome.
        assert!(job.time_limit_signaled_at.is_none());
    }

    // Reproduces the two steps report_job_status performs (validate the wire
    // report, then node_complete) since ControllerService can't be built here.
    // A signaled job's report (Completed, exit_code=0, signal=9) must be accepted
    // and rederived to Failed / exit_signal=9 / RaisedSignal.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rpc_path_signaled_completion_accepted_and_rederived_failed() {
        // Step 1: validate the wire report (Completed, exit_code=0) — must pass.
        JobState::validate_completion_report_state(JobState::Completed, 0)
            .expect("agent (Completed, exit_code=0) signaled report must pass RPC validation");

        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 8, 16000);

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("rpc-signal-job")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into()],
            resources: scalar_alloc(6, 12000),
            per_node_alloc: per_node_for(&["n1"], scalar_alloc(6, 12000)),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        // Step 2: the call the RPC makes after validation (wire state dropped).
        cm.node_complete(1, "n1", 0, 9, 0).unwrap();

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Failed);
        assert_eq!(job.exit_code, Some(0));
        assert_eq!(job.exit_signal, 9);
        assert_eq!(job.pending_reason, PendingReason::RaisedSignal);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_complete_sets_nonzero_exit_reason() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 8, 16000);

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("exit-job")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into()],
            resources: scalar_alloc(6, 12000),
            per_node_alloc: per_node_for(&["n1"], scalar_alloc(6, 12000)),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        cm.node_complete(1, "n1", 42, 0, 0).unwrap();
        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Failed);
        assert_eq!(job.exit_code, Some(42));
        assert_eq!(job.exit_signal, 0);
        // No srun steps ran, so DerivedExitCode is 0 (Slurm parity) — the batch
        // exit (42) surfaces as ExitCode, not DerivedExitCode.
        assert_eq!(job.derived_exit_code, 0);
        assert_eq!(job.pending_reason, PendingReason::NonZeroExitCode);
    }

    // A completion report from a superseded run (older epoch) must be dropped,
    // not fail the re-dispatched run. Reproduces the preempt-requeue race.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_complete_drops_stale_run_report() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("stale-job")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        // Re-dispatched run: current epoch is 2.
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into()],
            resources: scalar_alloc(6, 12000),
            per_node_alloc: per_node_for(&["n1"], scalar_alloc(6, 12000)),
            srun_step_dispatch: false,
            run_attempt: 2,
        });

        // Stale SIGKILL report from epoch 1 must be ignored.
        let res = cm.node_complete(1, "n1", 0, 9, 1).unwrap();
        assert!(matches!(res, NodeCompleteResult::StaleReport));
        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Running);

        // Current-epoch report is applied normally.
        cm.node_complete(1, "n1", 0, 9, 2).unwrap();
        assert_eq!(cm.get_job(1).unwrap().state, JobState::Failed);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_job_while_completing() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        for name in ["n1", "n2", "n3"] {
            register_node(&cm, name, 8, 16000);
        }

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("cancel-while-cg")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into(), "n2".into(), "n3".into()],
            resources: scalar_alloc(6, 12000),
            per_node_alloc: per_node_for(&["n1", "n2", "n3"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });

        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n1".into(),
            exit_code: 0,
            signal: 0,
        });

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Completing);
        assert_eq!(job.node_completions.len(), 1);
        assert_eq!(cm.get_node("n1").unwrap().alloc_resources.cpus, 0);
        assert!(cm.get_node("n2").unwrap().alloc_resources.cpus > 0);

        cm.cancel_job(1, "testuser").unwrap();
        settle(&cm, 1, JobState::Cancelled);

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Cancelled);
        assert_eq!(job.exit_code, Some(-1));
        assert!(job.node_completions.is_empty());
        for name in ["n1", "n2", "n3"] {
            assert_eq!(
                cm.get_node(name).unwrap().alloc_resources.cpus,
                0,
                "node {name} should be deallocated after cancel"
            );
        }

        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n2".into(),
            exit_code: 0,
            signal: 0,
        });

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_complete_returns_already_terminal_after_cancel() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        for name in ["n1", "n2", "n3"] {
            register_node(&cm, name, 8, 16000);
        }

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("nc-after-cancel")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["n1".into(), "n2".into(), "n3".into()],
            resources: scalar_alloc(6, 12000),
            per_node_alloc: per_node_for(&["n1", "n2", "n3"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });
        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: 1,
            node_name: "n1".into(),
            exit_code: 0,
            signal: 0,
        });

        cm.cancel_job(1, "testuser").unwrap();
        settle(&cm, 1, JobState::Cancelled);

        let result = cm.node_complete(1, "n2", 0, 0, 0).unwrap();
        assert_eq!(result, NodeCompleteResult::AlreadyTerminal);
        assert_eq!(cm.get_job(1).unwrap().state, JobState::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_metrics_track_lifecycle() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        assert_eq!(cm.job_metrics(), JobMetricsSnapshot::default());

        register_node(&cm, "worker1", 8, 16000);
        let job_id = submit_and_wait(&cm, basic_spec("metrics-job"));

        let m = cm.job_metrics();
        assert_eq!(m.total, 1);
        assert_eq!(m.count_state(JobState::Pending), 1);

        let resources = scalar_alloc(4, 8192);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        let m = cm.job_metrics();
        assert_eq!(m.count_state(JobState::Running), 1);
        assert_eq!(m.running_cpus, 4);
        assert_eq!(m.running_memory_bytes, 8192 * 1024 * 1024);

        cm.complete_job(job_id, 0, JobState::Completed).unwrap();
        settle(&cm, job_id, JobState::Completed);

        let m = cm.job_metrics();
        assert_eq!(m.count_state(JobState::Completed), 1);
        assert_eq!(m.running_cpus, 0);

        // Snapshot matches a full scan of the job map.
        let expected = JobMetricsSnapshot::collect(cm.get_jobs(&JobFilter::default()).iter());
        assert_eq!(cm.job_metrics(), expected);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_metrics_track_lifecycle() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        assert_eq!(cm.node_metrics(), NodeMetricsSnapshot::default());

        register_node(&cm, "worker1", 8, 16000);
        register_node(&cm, "worker2", 8, 16000);

        let m = cm.node_metrics();
        assert_eq!(m.total, 2);
        assert_eq!(m.total_cpus, 16);
        assert_eq!(m.alloc_cpus, 0);
        assert_eq!(m.per_node.len(), 2);
        assert_eq!(m.per_node[0].name, "worker1");
        assert_eq!(m.per_node[1].name, "worker2");

        let job_id = submit_and_wait(&cm, basic_spec("node-metrics-job"));
        let resources = scalar_alloc(4, 8192);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        let m = cm.node_metrics();
        assert_eq!(m.alloc_cpus, 4);
        let w1 = m.per_node.iter().find(|n| n.name == "worker1").unwrap();
        assert_eq!(w1.alloc_cpus, 4);

        cm.complete_job(job_id, 0, JobState::Completed).unwrap();
        settle(&cm, job_id, JobState::Completed);

        let m = cm.node_metrics();
        assert_eq!(m.alloc_cpus, 0);

        // Snapshot matches a full scan of the node map.
        let expected = NodeMetricsSnapshot::collect(cm.get_nodes().iter());
        assert_eq!(cm.node_metrics(), expected);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_job() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let job_id = submit_and_wait(&cm, basic_spec("cancel-me"));
        cm.cancel_job(job_id, "testuser").unwrap();
        settle(&cm, job_id, JobState::Cancelled);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Cancelled);
    }

    // A cancelled job keeps its reported path (unlike requeue): the file was
    // created there, so scontrol should still point at it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_preserves_output_path() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("cancel-me"));

        let alloc = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            alloc.clone(),
            per_node_for(&["n1"], alloc),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);

        cm.set_job_output_paths(id, "/tmp/spur.out".into(), "/tmp/spur.out".into());
        cm.cancel_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Cancelled);

        let job = cm.get_job(id).unwrap();
        assert_eq!(job.actual_stdout_path.as_deref(), Some("/tmp/spur.out"));
        assert_eq!(job.actual_stderr_path.as_deref(), Some("/tmp/spur.out"));
    }

    /// Drive a fresh job all the way to RUNNING on `node`, returning its id.
    fn run_job_on(cm: &ClusterManager, name: &str, node: &str) -> JobId {
        let job_id = submit_and_wait(cm, basic_spec(name));
        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec![node.into()],
            resources.clone(),
            per_node_for(&[node], resources),
        )
        .unwrap();
        settle(cm, job_id, JobState::Running);
        job_id
    }

    /// Drive an already-submitted job to RUNNING on `node`.
    fn start_on_node(cm: &ClusterManager, job_id: JobId, node: &str) -> JobId {
        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec![node.into()],
            resources.clone(),
            per_node_for(&[node], resources),
        )
        .unwrap();
        settle(cm, job_id, JobState::Running);
        job_id
    }

    fn run_qos_job_for(cm: &ClusterManager, user: &str, qos: &str, node: &str) -> JobId {
        let mut spec = basic_spec("held");
        spec.user = user.into();
        spec.qos = Some(qos.into());
        start_on_node(cm, submit_and_wait(cm, spec), node)
    }

    fn run_account_job_for(cm: &ClusterManager, user: &str, account: &str, node: &str) -> JobId {
        let mut spec = basic_spec("held");
        spec.user = user.into();
        spec.account = Some(account.into());
        start_on_node(cm, submit_and_wait(cm, spec), node)
    }

    fn capped_qos(name: &str, limits: spur_core::accounting::QosLimits) -> Qos {
        Qos {
            name: name.into(),
            limits,
            ..Default::default()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assoc_mgr_info_reports_each_user_against_the_qos_cap() {
        // A user over MaxJobsPU with the cap in force. The record has to name
        // the breach, since nothing else surfaces it once the jobs are running.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        cm.qos_cache().insert(capped_qos(
            "cnt",
            spur_core::accounting::QosLimits {
                max_jobs_per_user: Some(1),
                ..Default::default()
            },
        ));
        run_qos_job_for(&cm, "alice", "cnt", "n1");
        run_qos_job_for(&cm, "alice", "cnt", "n1");
        run_qos_job_for(&cm, "bob", "cnt", "n1");

        let info = cm.assoc_mgr_info(None);
        assert_eq!(info.qos_records.len(), 1, "one record for the QOS itself");
        let record = &info.qos_records[0];
        assert_eq!(record.scope, "cnt");
        // Group figures belong to the scope, so they are stated once rather than
        // repeated per user.
        assert_eq!(record.grp_running_jobs, 3);
        assert_eq!(
            record.user_caps.as_ref().and_then(|c| c.max_jobs),
            Some(1),
            "a QOS caps every user the same way, so the scope states it once"
        );

        assert_eq!(record.users.len(), 2);
        let alice = &record.users[0];
        assert_eq!(alice.user, "alice");
        assert_eq!(alice.running_jobs, 2);
        assert_eq!(
            alice.exceeded_caps(),
            vec![spur_core::accounting::Cap::MaxJobs]
        );

        let bob = &record.users[1];
        assert_eq!(bob.running_jobs, 1);
        assert!(bob.exceeded_caps().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assoc_mgr_info_reports_a_qos_nobody_is_using() {
        // A cap on an idle QOS is where a misconfiguration hides, so the record and
        // its caps appear with no users under it rather than being omitted.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.qos_cache().insert(capped_qos(
            "idle",
            spur_core::accounting::QosLimits {
                max_jobs_per_user: Some(4),
                ..Default::default()
            },
        ));

        let info = cm.assoc_mgr_info(None);
        assert_eq!(info.qos_records.len(), 1);
        let record = &info.qos_records[0];
        assert_eq!(record.scope, "idle");
        assert!(record.users.is_empty());
        assert_eq!(record.grp_running_jobs, 0);
        assert_eq!(record.user_caps.as_ref().and_then(|c| c.max_jobs), Some(4));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assoc_mgr_info_keeps_a_qos_the_cache_no_longer_knows() {
        // A QOS deleted after its jobs queued is gone from the definitions but its
        // jobs are still holding resources, so dropping the record would hide them.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        cm.qos_cache()
            .insert(capped_qos("doomed", Default::default()));
        run_qos_job_for(&cm, "alice", "doomed", "n1");
        cm.qos_cache()
            .insert(capped_qos("other", Default::default()));

        let info = cm.assoc_mgr_info(None);
        let doomed = info
            .qos_records
            .iter()
            .find(|r| r.scope == "doomed")
            .expect("a QOS with running jobs stays reported");
        assert_eq!(doomed.grp_running_jobs, 1);
        assert_eq!(doomed.users.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assoc_mgr_info_counts_distinct_nodes_like_the_gate() {
        // Node TRES is distinct-node occupancy, not a sum of per-job node counts:
        // two jobs sharing one node hold one node, which is what the gate charges.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        cm.qos_cache().insert(capped_qos("cnt", Default::default()));
        run_qos_job_for(&cm, "alice", "cnt", "n1");
        run_qos_job_for(&cm, "alice", "cnt", "n1");

        let alice = &cm.assoc_mgr_info(None).qos_records[0].users[0];
        assert_eq!(alice.running_tres.get(TresType::Node), 1);
        // CPU follows each job's request rather than its allocation, which is the
        // figure the gate charges; two one-CPU jobs hold two.
        assert_eq!(alice.running_tres.get(TresType::Cpu), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assoc_mgr_info_filters_to_one_user_without_distorting_group_totals() {
        // An operator asking about one user still needs the scope's totals, or a
        // group cap cannot be judged from the filtered view.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        cm.qos_cache().insert(capped_qos("cnt", Default::default()));
        run_qos_job_for(&cm, "alice", "cnt", "n1");
        run_qos_job_for(&cm, "bob", "cnt", "n1");

        let info = cm.assoc_mgr_info(Some("bob"));
        let record = &info.qos_records[0];
        assert_eq!(record.users.len(), 1);
        assert_eq!(record.users[0].user, "bob");
        assert_eq!(record.users[0].running_jobs, 1);
        assert_eq!(record.grp_running_jobs, 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assoc_mgr_info_reports_usage_with_caps_it_could_not_read() {
        // Accounting is on and the QOS cache is warm, but the association cache
        // has not loaded, so its caps are genuinely missing — `limits_readable`
        // is what says so rather than the scope looking uncapped.
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.accounting.database_url = "postgres://unused-in-test".into();
        let cm = test_cluster_with_config(&dir, cfg).await;
        register_node(&cm, "n1", 64, 128000);
        cm.qos_cache().insert(capped_qos("cnt", Default::default()));
        run_qos_job_for(&cm, "alice", "cnt", "n1");

        assert!(!cm.association_cache().is_loaded());
        let info = cm.assoc_mgr_info(None);
        assert!(!info.limits_readable);
        let record = &info.qos_records[0];
        assert_eq!(record.users[0].running_jobs, 1);
        assert!(record.users[0].exceeded_caps().is_empty());
    }

    #[test]
    fn limits_readable_gates_on_accounting_and_each_cache() {
        // Accounting off: no caps to load, so a cold cache is readable, not a
        // fault. Accounting on: both caches must be warm, so a split cache still
        // reads as incomplete.
        assert!(limits_readable(false, false, false));
        assert!(limits_readable(true, true, true));
        assert!(!limits_readable(true, true, false));
        assert!(!limits_readable(true, false, true));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assoc_mgr_info_reads_limits_when_accounting_is_disabled() {
        // With accounting off there are no caps to load, so a cold cache is not a
        // fault: the banner must not stand at `LimitsReadable=NO` on a cluster
        // that simply has no limits.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        run_account_job_for(&cm, "alice", "tenant-a", "n1");

        let info = cm.assoc_mgr_info(None);
        assert!(info.limits_readable);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assoc_mgr_info_hides_scopes_a_filtered_user_has_no_part_in() {
        // A user-scoped read must not enumerate the cluster's QOS inventory: a QOS
        // the caller holds no work under is dropped, not returned empty.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        cm.qos_cache()
            .insert(capped_qos("mine", Default::default()));
        cm.qos_cache()
            .insert(capped_qos("theirs", Default::default()));
        run_qos_job_for(&cm, "alice", "mine", "n1");
        run_qos_job_for(&cm, "bob", "theirs", "n1");

        let info = cm.assoc_mgr_info(Some("alice"));
        let scopes: Vec<&str> = info.qos_records.iter().map(|r| r.scope.as_str()).collect();
        assert_eq!(scopes, vec!["mine"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assoc_mgr_info_keeps_qos_and_account_records_apart() {
        // A job with both lands in both sections, each carrying that hierarchy's own
        // caps: the account record shows the association's MaxJobs, not the QOS's.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        cm.qos_cache().insert(capped_qos(
            "cnt",
            spur_core::accounting::QosLimits {
                max_jobs_per_user: Some(9),
                ..Default::default()
            },
        ));
        cm.association_cache().insert_limits(
            "alice",
            "tenant-a",
            AccountLimits {
                max_running_jobs: Some(4),
                ..Default::default()
            },
        );
        let mut spec = basic_spec("both");
        spec.user = "alice".into();
        spec.qos = Some("cnt".into());
        spec.account = Some("tenant-a".into());
        start_on_node(&cm, submit_and_wait(&cm, spec), "n1");

        let info = cm.assoc_mgr_info(None);
        assert_eq!(info.qos_records.len(), 1);
        assert_eq!(info.assoc_records.len(), 1);
        assert_eq!(
            info.qos_records[0].users[0].caps.max_jobs,
            Some(9),
            "the QOS record carries the QOS's per-user cap"
        );
        let assoc = &info.assoc_records[0];
        assert_eq!(assoc.scope, "tenant-a");
        assert_eq!(assoc.users[0].caps.max_jobs, Some(4));
        // An association has no scope-wide per-user caps: each of its rows carries
        // its own, and it has no per-user TRES cap at all.
        assert!(assoc.user_caps.is_none());
        assert_eq!(assoc.users[0].caps.max_tres, None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assoc_mgr_info_reports_an_association_with_no_jobs() {
        // Association caps are per (user, account), so an idle association's user
        // has to be listed or its cap would be invisible.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.association_cache().insert_association("bob", "tenant-b");
        cm.association_cache().insert_limits(
            "bob",
            "tenant-b",
            AccountLimits {
                max_running_jobs: Some(3),
                ..Default::default()
            },
        );

        let info = cm.assoc_mgr_info(None);
        let record = info
            .assoc_records
            .iter()
            .find(|r| r.scope == "tenant-b")
            .expect("a defined association is reported before it is used");
        assert_eq!(record.users.len(), 1);
        assert_eq!(record.users[0].user, "bob");
        assert_eq!(record.users[0].running_jobs, 0);
        assert_eq!(record.users[0].caps.max_jobs, Some(3));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assoc_mgr_info_is_empty_without_definitions_or_scoped_work() {
        // Jobs with neither a QOS nor an account produce no records: there is no
        // scope to report them under.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        run_job_on(&cm, "plain", "n1");

        let info = cm.assoc_mgr_info(None);
        assert!(info.qos_records.is_empty());
        assert!(info.assoc_records.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_running_job_frees_nodes_and_repends() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "requeue-me", "worker1");
        assert_eq!(cm.node_metrics().alloc_cpus, 2);

        let outcome = cm
            .requeue_job_by_user(job_id, "testuser", false, false)
            .unwrap();
        assert_eq!(
            outcome.killed.len(),
            1,
            "a running job must be reported for agent kill"
        );
        settle(&cm, job_id, JobState::Pending);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert!(job.allocated_nodes.is_empty());
        assert_eq!(cm.node_metrics().alloc_cpus, 0, "nodes must be freed");
        assert_eq!(job.user_requeue_count, 1);
        assert_eq!(
            job.requeue_count, 0,
            "manual requeue must not touch the batch cap"
        );
        // A live job carries a short begin_time hold so the scheduler cannot
        // re-dispatch it into its own in-flight cancel.
        assert_eq!(job.pending_reason, PendingReason::BeginTime);
        let begin = job
            .spec
            .begin_time
            .expect("live requeue sets a begin_time hold");
        assert!(begin > Utc::now(), "begin_time hold must be in the future");
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "held-by-begin-time job must not be eligible yet"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_suspended_job_repends() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "suspend-requeue", "worker1");
        cm.suspend_job(job_id, "testuser").unwrap();
        settle(&cm, job_id, JobState::Suspended);

        let outcome = cm
            .requeue_job_by_user(job_id, "testuser", false, false)
            .unwrap();
        assert_eq!(outcome.killed.len(), 1);
        settle(&cm, job_id, JobState::Pending);

        let job = cm.get_job(job_id).unwrap();
        assert!(job.allocated_nodes.is_empty());
        assert_eq!(cm.node_metrics().alloc_cpus, 0, "nodes must be freed");
        assert_eq!(job.user_requeue_count, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_completed_job_repends_without_extra_finalize() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "completed-requeue", "worker1");
        cm.complete_job(job_id, 0, JobState::Completed).unwrap();
        settle(&cm, job_id, JobState::Completed);

        // An already-terminal job has no live processes to kill.
        let outcome = cm
            .requeue_job_by_user(job_id, "testuser", false, false)
            .unwrap();
        assert!(
            outcome.killed.is_empty(),
            "terminal job needs no agent kill"
        );
        assert_eq!(outcome.requeued, 1);
        settle(&cm, job_id, JobState::Pending);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.user_requeue_count, 1);
        assert!(job.end_time.is_none(), "re-pended job clears end_time");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_all_terminal_states_repend() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 32, 64000);

        // Every terminal state must re-pend, so the leader's acceptance and the
        // apply-side transition never disagree (no "requeued" print without effect).
        for state in [
            JobState::Failed,
            JobState::Timeout,
            JobState::NodeFail,
            JobState::OutOfMemory,
            JobState::Cancelled,
        ] {
            let job_id = run_job_on(&cm, &format!("term-{state:?}"), "worker1");
            cm.complete_job(job_id, -1, state).unwrap();
            settle(&cm, job_id, state);

            cm.requeue_job_by_user(job_id, "testuser", false, false)
                .unwrap();
            settle(&cm, job_id, JobState::Pending);
            assert_eq!(cm.get_job(job_id).unwrap().user_requeue_count, 1);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_terminal_job_does_not_double_free_shared_node() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let a = run_job_on(&cm, "share-a", "worker1");
        let _b = run_job_on(&cm, "share-b", "worker1");
        assert_eq!(cm.node_metrics().alloc_cpus, 4);

        cm.complete_job(a, 0, JobState::Completed).unwrap();
        settle(&cm, a, JobState::Completed);
        assert_eq!(
            cm.node_metrics().alloc_cpus,
            2,
            "A's slice freed once at completion"
        );

        // Requeuing already-terminal A must not free its slice a second time, or
        // B's accounting on the shared node would be corrupted.
        cm.requeue_job_by_user(a, "testuser", false, false).unwrap();
        settle(&cm, a, JobState::Pending);
        assert_eq!(
            cm.node_metrics().alloc_cpus,
            2,
            "terminal requeue must not double-free the shared node"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_finalizes_only_live_jobs() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        // Live path finalizes once (REQUEUED); the terminal path emits none, so
        // a completed job is never accounted twice.
        let live = run_job_on(&cm, "finalize-live", "worker1");
        let resp = cm.apply_operation(&WalOperation::JobUserRequeue {
            job_id: live,
            hold: false,
            begin_time: None,
        });
        assert_eq!(resp.jobs_finalized.len(), 1);
        assert_eq!(resp.jobs_finalized[0].state, JobState::Requeued);

        let done = run_job_on(&cm, "finalize-done", "worker1");
        cm.complete_job(done, 0, JobState::Completed).unwrap();
        settle(&cm, done, JobState::Completed);
        let resp = cm.apply_operation(&WalOperation::JobUserRequeue {
            job_id: done,
            hold: false,
            begin_time: None,
        });
        assert!(
            resp.jobs_finalized.is_empty(),
            "an already-terminal job must not be finalized again"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_hold_parks_job_held() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "requeue-hold", "worker1");
        cm.requeue_job_by_user(job_id, "testuser", false, true)
            .unwrap();
        settle(&cm, job_id, JobState::Pending);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.priority, 0, "held job sits at priority 0");
        assert_eq!(job.pending_reason, PendingReason::Held);
        assert!(job.pending_reason.is_scheduling_hold());
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "held job must not be eligible for dispatch"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_rejects_interactive_jobs() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // Each of the three non-batch markers must be rejected (the check runs
        // before the state match, so a Pending allocation is enough to exercise
        // it). srun/salloc/--pty have no batch script to re-run.
        for (name, apply) in [
            (
                "srun",
                (|s: &mut JobSpec| s.srun_job = true) as fn(&mut JobSpec),
            ),
            ("interactive", |s: &mut JobSpec| s.interactive = true),
            ("pty", |s: &mut JobSpec| s.pty = true),
        ] {
            let mut spec = basic_spec(name);
            apply(&mut spec);
            let job_id = submit_and_wait(&cm, spec);
            let err = cm
                .requeue_job_by_user(job_id, "testuser", false, false)
                .expect_err("interactive jobs cannot be requeued");
            assert!(err.to_string().contains("interactive"), "{name}: {err}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_hold_then_release_reschedules() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "hold-release", "worker1");
        cm.requeue_job_by_user(job_id, "testuser", false, true)
            .unwrap();
        settle(&cm, job_id, JobState::Pending);
        assert!(cm
            .get_job(job_id)
            .unwrap()
            .pending_reason
            .is_scheduling_hold());

        // Releasing the held requeue makes it schedulable again.
        cm.release_job(job_id).unwrap();
        wait_for("released", || {
            cm.get_job(job_id)
                .is_some_and(|j| !j.pending_reason.is_scheduling_hold())
        });
        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.priority, DEFAULT_PRIORITY);
        assert!(cm.pending_jobs().iter().any(|j| j.job_id == job_id));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_preserves_later_user_begin() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        // A user --begin further out than the requeue's own hold must win, so
        // the requeue never shortens the operator's constraint.
        let user_begin = Utc::now() + chrono::Duration::hours(1);
        let mut spec = basic_spec("user-begin-requeue");
        spec.begin_time = Some(user_begin);
        let job_id = submit_and_wait(&cm, spec);
        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        cm.requeue_job_by_user(job_id, "testuser", false, false)
            .unwrap();
        settle(&cm, job_id, JobState::Pending);
        assert_eq!(
            cm.get_job(job_id).unwrap().spec.begin_time,
            Some(user_begin),
            "user --begin beyond the requeue hold must be preserved"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_preserves_run_attempt_fencing() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        // First run is epoch 1.
        let job_id = run_job_on(&cm, "epoch", "worker1");
        assert_eq!(cm.get_job(job_id).unwrap().run_attempt, 1);

        cm.requeue_job_by_user(job_id, "testuser", false, false)
            .unwrap();
        settle(&cm, job_id, JobState::Pending);

        // Re-dispatch bumps the epoch to 2 (requeue must not have reset it).
        let alloc = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            alloc.clone(),
            per_node_for(&["worker1"], alloc),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);
        assert_eq!(cm.get_job(job_id).unwrap().run_attempt, 2);

        // A late completion from the killed run (epoch 1) must be dropped so the
        // old cancel can never finalize the fresh run.
        let res = cm.node_complete(job_id, "worker1", 0, 9, 1).unwrap();
        assert!(matches!(res, NodeCompleteResult::StaleReport));
        assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Running);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_rejects_already_pending() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let job_id = submit_and_wait(&cm, basic_spec("pending"));

        let err = cm
            .requeue_job_by_user(job_id, "testuser", false, false)
            .expect_err("a pending job cannot be requeued");
        assert!(err.to_string().contains("already pending"), "got: {err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_rejects_wrong_user() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);
        let job_id = run_job_on(&cm, "owned-by-testuser", "worker1");

        let err = cm
            .requeue_job_by_user(job_id, "bob", false, false)
            .expect_err("a non-owner non-admin must be denied");
        assert!(
            err.downcast_ref::<spur_core::auth::AuthError>().is_some(),
            "got: {err}"
        );
        // an admin may requeue any job.
        cm.requeue_job_by_user(job_id, "root", true, false).unwrap();
        settle(&cm, job_id, JobState::Pending);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_array_parent_fans_out_to_all_tasks() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let mut spec = basic_spec("array-requeue");
        spec.array_spec = Some("0-1".into());
        let parent = cm.submit_job(spec).unwrap().job_id;

        // Collect the task ids and run each to completion so they are requeue-able.
        wait_for("array tasks applied", || {
            cm.get_jobs(&JobFilter::default())
                .iter()
                .filter(|j| j.spec.array_job_id == Some(parent))
                .count()
                == 2
        });
        let task_ids: Vec<JobId> = cm
            .get_jobs(&JobFilter::default())
            .iter()
            .filter(|j| j.spec.array_job_id == Some(parent))
            .map(|j| j.job_id)
            .collect();
        for &id in &task_ids {
            let resources = scalar_alloc(2, 4000);
            cm.start_job(
                id,
                vec!["worker1".into()],
                resources.clone(),
                per_node_for(&["worker1"], resources),
            )
            .unwrap();
            settle(&cm, id, JobState::Running);
            cm.complete_job(id, 0, JobState::Completed).unwrap();
            settle(&cm, id, JobState::Completed);
        }

        // Requeuing the bare array-parent id fans out to every task.
        let outcome = cm
            .requeue_job_by_user(parent, "testuser", false, false)
            .unwrap();
        assert_eq!(outcome.requeued, task_ids.len() as u32);
        assert!(outcome.skipped.is_empty());
        for &id in &task_ids {
            settle(&cm, id, JobState::Pending);
            assert_eq!(cm.get_job(id).unwrap().user_requeue_count, 1);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_array_all_pending_errors() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // No node registered, so both tasks stay Pending; requeuing them is
        // rejected per task, and with none requeue-able the whole call errors.
        let mut spec = basic_spec("array-all-pending");
        spec.array_spec = Some("0-1".into());
        let parent = cm.submit_job(spec).unwrap().job_id;
        wait_for("array tasks applied", || {
            cm.get_jobs(&JobFilter::default())
                .iter()
                .filter(|j| j.spec.array_job_id == Some(parent))
                .count()
                == 2
        });

        let err = cm
            .requeue_job_by_user(parent, "testuser", false, false)
            .expect_err("all-pending array requeue must error");
        assert!(err.to_string().contains("no tasks of array"), "got: {err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_rejects_completing_job() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "completing", "worker1");
        cm.apply_operation(&WalOperation::job_state_change(
            job_id,
            JobState::Running,
            JobState::Completing,
        ));
        settle(&cm, job_id, JobState::Completing);

        let err = cm
            .requeue_job_by_user(job_id, "testuser", false, false)
            .expect_err("a completing job cannot be requeued");
        assert!(err.to_string().contains("completing"), "got: {err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_array_partial_reports_skipped_tasks() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let mut spec = basic_spec("array-partial");
        spec.array_spec = Some("0-1".into());
        let parent = cm.submit_job(spec).unwrap().job_id;
        wait_for("array tasks applied", || {
            cm.get_jobs(&JobFilter::default())
                .iter()
                .filter(|j| j.spec.array_job_id == Some(parent))
                .count()
                == 2
        });
        let mut task_ids: Vec<JobId> = cm
            .get_jobs(&JobFilter::default())
            .iter()
            .filter(|j| j.spec.array_job_id == Some(parent))
            .map(|j| j.job_id)
            .collect();
        task_ids.sort_unstable();

        // Run only the first task; leave the second Pending so the fan-out
        // requeues one and skips the other ("already pending").
        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            task_ids[0],
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, task_ids[0], JobState::Running);

        let outcome = cm
            .requeue_job_by_user(parent, "testuser", false, false)
            .unwrap();
        assert_eq!(outcome.requeued, 1, "only the running task is requeued");
        assert_eq!(
            outcome.skipped.len(),
            1,
            "the pending task is surfaced as skipped"
        );
        assert!(
            outcome.skipped[0].contains("already pending"),
            "got: {:?}",
            outcome.skipped
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_apply_is_idempotent_on_replay() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);
        let job_id = run_job_on(&cm, "replay", "worker1");

        cm.apply_operation(&WalOperation::JobUserRequeue {
            job_id,
            hold: false,
            begin_time: None,
        });
        assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Pending);
        assert_eq!(cm.get_job(job_id).unwrap().user_requeue_count, 1);

        // Re-applying the same entry (follower catch-up / replay) is a NoOp:
        // the job is already Pending, so the counter must not double-bump.
        cm.apply_operation(&WalOperation::JobUserRequeue {
            job_id,
            hold: false,
            begin_time: None,
        });
        assert_eq!(
            cm.get_job(job_id).unwrap().user_requeue_count,
            1,
            "replay must not double-count a requeue"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn user_requeue_never_holds_at_max_batch_requeue() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);
        let job_id = run_job_on(&cm, "repeat-requeue", "worker1");

        let max = cm.config().controller.max_batch_requeue;
        for n in 0..(max + 2) {
            // Re-dispatch then requeue again; the batch cap must never trigger.
            if cm.get_job(job_id).unwrap().state == JobState::Pending {
                let resources = scalar_alloc(2, 4000);
                cm.start_job(
                    job_id,
                    vec!["worker1".into()],
                    resources.clone(),
                    per_node_for(&["worker1"], resources),
                )
                .unwrap();
                settle(&cm, job_id, JobState::Running);
            }
            cm.requeue_job_by_user(job_id, "testuser", false, false)
                .unwrap();
            settle(&cm, job_id, JobState::Pending);
            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.user_requeue_count, n + 1);
            assert_eq!(job.requeue_count, 0);
            assert_ne!(
                job.pending_reason,
                PendingReason::JobHoldMaxRequeue,
                "manual requeue must never trip the batch-requeue hold"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_job_requeue_returns_job_to_pending_and_frees_nodes() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        // --requeue not set on the spec: requeue preempt mode must still
        // return the job to the queue rather than stranding it in PREEMPTED.
        let job_id = run_job_on(&cm, "preempt-requeue", "worker1");
        assert_eq!(cm.node_metrics().alloc_cpus, 2);

        let outcome = cm
            .preempt_job_with_provenance(job_id, PreemptMode::Requeue, Some(99), None)
            .unwrap();
        assert_eq!(outcome, PreemptOutcome::Killed);
        settle(&cm, job_id, JobState::Pending);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert!(job.allocated_nodes.is_empty());
        assert_eq!(cm.node_metrics().alloc_cpus, 0, "nodes must be freed");

        // The requeue must carry a future begin_time hold so the scheduler
        // cannot re-dispatch the job into its own in-flight preemption cancel.
        let begin = job
            .spec
            .begin_time
            .expect("requeue must set a begin_time hold");
        assert!(begin > Utc::now(), "begin_time hold must be in the future");
        assert_eq!(job.pending_reason, PendingReason::Preempted);
        assert!(
            job.pending_reason_desc
                .as_deref()
                .unwrap_or("")
                .contains("preempted by job 99"),
            "pending_reason_desc must name the preempting job"
        );
        assert_eq!(job.preempted_by, Some(99));
        assert_eq!(job.preempt_mode.as_deref(), Some("Requeue"));

        // While the hold is active the job is excluded from scheduling.
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "held job must not be eligible for dispatch"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_requeue_preserves_later_user_begin() {
        // A user --begin further out than the preemption hold must win: the
        // requeue must not shorten the user's constraint.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let user_begin = Utc::now() + chrono::Duration::hours(1);
        let mut spec = basic_spec("user-begin");
        spec.begin_time = Some(user_begin);
        let job_id = submit_and_wait(&cm, spec);
        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        cm.preempt_job_with_provenance(job_id, PreemptMode::Requeue, Some(99), None)
            .unwrap();
        settle(&cm, job_id, JobState::Pending);

        assert_eq!(
            cm.get_job(job_id).unwrap().spec.begin_time,
            Some(user_begin),
            "user --begin beyond the hold must be preserved"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_requeue_hold_reason_survives_pending_reason_passes() {
        // The BeginTime hold reason must not be clobbered by the pending-reason
        // maintenance passes while the hold is still active.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "hold-reason", "worker1");
        cm.preempt_job_with_provenance(job_id, PreemptMode::Requeue, Some(99), None)
            .unwrap();
        settle(&cm, job_id, JobState::Pending);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::Preempted
        );

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::Preempted,
            "refresh_pending_reasons must not clobber an active Preempted hold"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_requeue_hold_expires_and_job_reschedules() {
        // Once the hold lapses the job must become eligible and actually run
        // again — the hold must not permanently strand it.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "hold-expiry", "worker1");
        cm.preempt_job_with_provenance(job_id, PreemptMode::Requeue, Some(99), None)
            .unwrap();
        settle(&cm, job_id, JobState::Pending);

        // Simulate the hold having elapsed by moving begin_time into the past,
        // exactly as the wall clock would after the hold window.
        {
            let mut jobs = cm.jobs.write();
            let job = jobs.get_mut(&job_id).unwrap();
            job.spec.begin_time = Some(Utc::now() - chrono::Duration::seconds(1));
        }

        assert!(
            cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "job must be eligible again once the hold lapses"
        );

        // And it can be started again (not stranded).
        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);
    }

    // ── requeue-after-launch-failure backoff ─────────────────────

    /// Let an active hold lapse, exactly as the wall clock would, so the next
    /// dispatch attempt can proceed without sleeping in the test.
    fn lapse_hold(cm: &ClusterManager, job_id: JobId) {
        let mut jobs = cm.jobs.write();
        let job = jobs.get_mut(&job_id).unwrap();
        job.spec.begin_time = Some(Utc::now() - chrono::Duration::seconds(1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn launch_failure_requeue_holds_job_and_reports_reason() {
        // The dispatch-failure requeue must defer the job and say why. Without
        // the hold the scheduler re-dispatches on the very next tick, and for a
        // node-local fault that means the same broken node again.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "launch-fail", "worker1");
        assert_eq!(cm.node_metrics().alloc_cpus, 2);

        cm.requeue_job(job_id).unwrap();
        settle(&cm, job_id, JobState::Pending);

        let job = cm.get_job(job_id).unwrap();
        assert!(job.allocated_nodes.is_empty());
        assert_eq!(cm.node_metrics().alloc_cpus, 0, "nodes must be freed");
        assert_eq!(job.requeue_count, 1);
        assert_eq!(
            job.pending_reason,
            PendingReason::JobLaunchFailure,
            "squeue must show why the job is waiting"
        );
        assert!(
            job.spec.begin_time.is_some_and(|begin| begin > Utc::now()),
            "requeue must carry a future hold"
        );
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "held job must not be eligible for immediate re-dispatch"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn requeueing_a_job_that_never_started_is_a_no_op() {
        // Pending -> Failed is illegal and Pending -> Pending applies as a NoOp,
        // so the two proposals would be silently discarded. Bailing out keeps the
        // caller from logging a hold that was never applied.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = submit_and_wait(&cm, basic_spec("never-started"));
        cm.requeue_job(job_id).unwrap();

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.requeue_count, 0, "no attempt was consumed");
        assert!(job.spec.begin_time.is_none(), "no hold was applied");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_pending_job_at_the_requeue_cap_is_held_not_ignored() {
        // The cap has to be checked before the never-started bail-out. A job
        // that exhausts its budget while Pending would otherwise be dropped on
        // the floor and re-dispatched at full priority every scheduler tick.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);
        let max = cm.config().controller.max_batch_requeue;

        let job_id = submit_and_wait(&cm, basic_spec("pending-at-cap"));
        cm.jobs.write().get_mut(&job_id).unwrap().requeue_count = max;

        cm.requeue_job(job_id).unwrap();

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.pending_reason, PendingReason::JobHoldMaxRequeue);
        assert_eq!(job.priority, 0, "a held job must not outrank live work");
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "the hold must take the job out of scheduling"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn draining_a_node_for_a_launch_failure_still_lets_the_job_requeue() {
        // The agent drains the node on a spool fault. That drain must not
        // finalize the job: it still has to reach Pending so the scheduler can
        // retry it somewhere healthy.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);
        register_node(&cm, "worker2", 8, 16000);

        let job_id = run_job_on(&cm, "spool-fault", "worker1");

        let (state, _) = cm
            .drain_node("worker1", Some("launch failed: ENOSPC".into()), None)
            .unwrap();
        assert_eq!(state, NodeState::Draining, "the job still holds the node");

        cm.requeue_job(job_id).unwrap();
        settle(&cm, job_id, JobState::Pending);

        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::JobLaunchFailure
        );
        let node = cm.get_node("worker1").unwrap();
        assert!(
            node.state.is_admin_hold(),
            "the broken node must stay excluded from scheduling"
        );
        assert_eq!(
            node.state_reason.as_deref(),
            Some("launch failed: ENOSPC"),
            "operators need the fault text"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_completion_report_for_a_launch_failure_strands_the_job() {
        // Why the agent drains instead of reporting a completion: a completion
        // finalizes the job, and requeue_job refuses terminal jobs, so the
        // launch failure would never be retried anywhere.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "stranded", "worker1");
        cm.node_complete(job_id, "worker1", -1, 0, 0).unwrap();
        wait_for("job reaches a terminal state", || {
            cm.get_job(job_id).is_some_and(|j| j.state.is_terminal())
        });

        cm.requeue_job(job_id).unwrap();

        let job = cm.get_job(job_id).unwrap();
        assert!(
            job.state.is_terminal(),
            "the completion report won the race"
        );
        assert_eq!(job.requeue_count, 0, "a terminal job is never retried");
    }

    #[test]
    fn the_backoff_doubles_each_attempt_up_to_the_cap() {
        // The schedule itself, checked as arithmetic. With the default
        // interval_secs = 1 the base is 5s, so the five attempts allowed by
        // max_batch_requeue = 5 wait 5, 10, 20, 40 and 80 seconds: 155s in
        // total instead of burning the budget in ~5s.
        let uncapped = spur_core::config::MAX_LAUNCH_BACKOFF_SECS;
        for (attempt, expected) in [5u64, 10, 20, 40, 80].into_iter().enumerate() {
            assert_eq!(
                launch_backoff_secs(1, uncapped, attempt as u32),
                expected,
                "attempt {attempt}"
            );
        }

        // A cap clamps the doubling without stopping it from reaching the cap.
        assert_eq!(launch_backoff_secs(1, 12, 0), 5);
        assert_eq!(launch_backoff_secs(1, 12, 1), 10);
        assert_eq!(launch_backoff_secs(1, 12, 2), 12);

        // A slow scheduler waits at least two of its own cycles plus slack, and
        // a zero interval still gets the 5s floor rather than no backoff at all.
        assert_eq!(launch_backoff_secs(10, uncapped, 0), 23);
        assert_eq!(launch_backoff_secs(0, uncapped, 0), 5);

        // An operator-supplied max_batch_requeue large enough to overflow the
        // shift must saturate at the cap, not panic the controller.
        assert_eq!(launch_backoff_secs(1, 300, u32::MAX), 300);
        assert_eq!(launch_backoff_secs(1, uncapped, u32::MAX), uncapped);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn launch_failure_backoff_defers_every_attempt_until_the_budget_runs_out() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = submit_and_wait(&cm, basic_spec("backoff-schedule"));

        for attempt in 0..cm.config().controller.max_batch_requeue {
            let resources = scalar_alloc(2, 4000);
            cm.start_job(
                job_id,
                vec!["worker1".into()],
                resources.clone(),
                per_node_for(&["worker1"], resources),
            )
            .unwrap();
            settle(&cm, job_id, JobState::Running);

            cm.requeue_job(job_id).unwrap();
            settle(&cm, job_id, JobState::Pending);

            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.requeue_count, attempt + 1);
            assert!(
                job.spec.begin_time.is_some_and(|begin| begin > Utc::now()),
                "attempt {attempt} must defer the retry"
            );
            assert!(
                !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
                "attempt {attempt} must not leave the job immediately eligible"
            );

            lapse_hold(&cm, job_id);
        }

        // Budget exhausted: the next failure holds the job for an operator.
        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);
        cm.requeue_job(job_id).unwrap();
        settle(&cm, job_id, JobState::Pending);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::JobHoldMaxRequeue
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn launch_failure_requeue_preserves_later_user_begin() {
        // The backoff must never shorten a user's --begin constraint.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let user_begin = Utc::now() + chrono::Duration::hours(1);
        let mut spec = basic_spec("user-begin-launch-fail");
        spec.begin_time = Some(user_begin);
        let job_id = submit_and_wait(&cm, spec);
        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        cm.requeue_job(job_id).unwrap();
        settle(&cm, job_id, JobState::Pending);

        assert_eq!(
            cm.get_job(job_id).unwrap().spec.begin_time,
            Some(user_begin),
            "user --begin beyond the hold must be preserved"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn launch_failure_reason_survives_both_pending_reason_passes() {
        // JobLaunchFailure must outlive every pass that recomputes
        // pending_reason. The predicate lives in four places; miss one and the
        // reason is clobbered back to Resources before an operator sees it.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "reason-survives", "worker1");
        cm.requeue_job(job_id).unwrap();
        settle(&cm, job_id, JobState::Pending);

        // Both guard sites in refresh_pending_reasons.
        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::JobLaunchFailure,
            "refresh_pending_reasons must not clobber an active hold"
        );

        // The guard site in update_pending_reasons. An empty cluster_state would
        // otherwise force Resources/NodeDown.
        let empty_state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &[],
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        let snapshot = cm.get_job(job_id).unwrap();
        cm.update_pending_reasons(&[&snapshot], &empty_state);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::JobLaunchFailure,
            "update_pending_reasons must not clobber an active hold"
        );

        // The guard site in cancel_unsatisfiable_dependency_jobs.
        cm.cancel_unsatisfiable_dependency_jobs();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::JobLaunchFailure,
            "the dependency pass must not clobber an active hold"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_user_begin_job_still_reports_what_blocks_it() {
        // A --begin job carries no reason explaining its own hold, so the wait
        // reason passes must still tag it. Skipping every held job would leave
        // squeue showing None for as long as the user asked to wait.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let mut spec = basic_spec("begin-blocked");
        spec.begin_time = Some(Utc::now() + chrono::Duration::hours(1));
        let job_id = submit_and_wait(&cm, spec);
        assert!(cm.get_job(job_id).unwrap().is_begin_held(Utc::now()));

        let empty_state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &[],
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        let snapshot = cm.get_job(job_id).unwrap();
        cm.update_pending_reasons(&[&snapshot], &empty_state);

        let reason = cm.get_job(job_id).unwrap().pending_reason;
        assert_ne!(
            reason,
            PendingReason::None,
            "a held --begin job must still say what blocks it"
        );
        assert!(
            !reason.explains_begin_hold(),
            "the hold is not the blocker here, the empty cluster is"
        );
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "tagging a reason must not make a held job eligible"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn launch_failure_hold_expires_and_job_reschedules() {
        // The hold defers the job, it must not strand it.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "hold-lapses", "worker1");
        cm.requeue_job(job_id).unwrap();
        settle(&cm, job_id, JobState::Pending);
        lapse_hold(&cm, job_id);

        assert!(
            cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "job must be eligible again once the hold lapses"
        );

        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn replayed_backoff_requeue_does_not_double_count_or_drift() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("backoff-replay")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: -1,
            state: JobState::Failed,
        });

        let hold = Utc::now() + chrono::Duration::seconds(40);
        let requeue = WalOperation::job_state_change_backoff_pending(
            1,
            JobState::Failed,
            PendingReason::JobLaunchFailure,
            hold,
        );
        cm.apply_operation(&requeue);

        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.requeue_count, 1);
        assert_eq!(job.pending_reason, PendingReason::JobLaunchFailure);
        assert_eq!(job.spec.begin_time, Some(hold));

        // Replay the identical entry: the job is already Pending, so this is a
        // NoOp. Followers and WAL recovery must not compound the hold.
        cm.apply_operation(&requeue);
        let job = cm.get_job(1).unwrap();
        assert_eq!(
            job.requeue_count, 1,
            "replayed requeue must not double-count"
        );
        assert_eq!(
            job.spec.begin_time,
            Some(hold),
            "replayed hold instant must not drift"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_job_cancel_terminates_and_frees_nodes() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "preempt-cancel", "worker1");
        let outcome = cm
            .preempt_job_with_provenance(job_id, PreemptMode::Cancel, Some(99), None)
            .unwrap();
        assert_eq!(outcome, PreemptOutcome::Killed);
        settle(&cm, job_id, JobState::Cancelled);

        let job = cm.get_job(job_id).unwrap();
        // Preempted is reported to accounting (jobs_finalized); the job's
        // live state is Cancelled so dependents and arrays can resolve.
        assert_eq!(job.state, JobState::Cancelled);
        assert_eq!(cm.node_metrics().alloc_cpus, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_preempt_cancel_is_atomic_and_replay_deterministic() {
        // JobPreemptCancel transitions Running → Preempted, frees nodes, and
        // fires jobs_finalized in one WAL entry. Replay on a non-running job
        // is a NoOp: no double-free, no re-finalize.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("preempt-cancel-replay")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["worker1".into()],
            resources: alloc.clone(),
            per_node_alloc: per_node_for(&["worker1"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });
        assert_eq!(cm.get_node("worker1").unwrap().alloc_resources.cpus, 2);

        // Apply with explicit provenance to verify field propagation.
        let resp = cm.apply_operation(&WalOperation::JobPreemptCancel {
            job_id: 1,
            preempted_by: Some(42),
            preempt_qos: Some("highprio".into()),
        });

        assert_eq!(resp.jobs_finalized.len(), 1);
        // Accounting sees Preempted; live state is Cancelled (terminal).
        assert_eq!(resp.jobs_finalized[0].state, JobState::Preempted);
        assert_eq!(resp.jobs_finalized[0].exit_code, -1);
        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Cancelled);
        assert_eq!(job.exit_code, Some(-1));
        assert_eq!(job.preempted_by, Some(42));
        assert_eq!(job.preempt_mode.as_deref(), Some("Cancel"));
        assert_eq!(job.preempt_qos.as_deref(), Some("highprio"));
        // allocated_nodes preserved (epilog reads NodeList after finalize).
        assert!(!job.allocated_nodes.is_empty());
        assert_eq!(cm.get_node("worker1").unwrap().alloc_resources.cpus, 0);

        // Replay: job is already Cancelled (not Running) → NoOp.
        let replay = cm.apply_operation(&WalOperation::JobPreemptCancel {
            job_id: 1,
            preempted_by: Some(42),
            preempt_qos: Some("highprio".into()),
        });
        assert!(
            replay.jobs_finalized.is_empty(),
            "replayed preempt-cancel must not re-finalize"
        );
        assert_eq!(
            cm.get_node("worker1").unwrap().alloc_resources.cpus,
            0,
            "replayed preempt-cancel must not double-free"
        );
    }

    fn budget_qos(cap: u32) -> Qos {
        Qos {
            name: "budget".into(),
            limits: spur_core::accounting::QosLimits {
                grp_wall_minutes: Some(cap),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn grp_wall_budget_blocks_scheduling_and_tags_the_slurm_reason() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        cm.qos_cache().insert(budget_qos(600));
        cm.grp_wall_cache()
            .seed(HashMap::from([("budget".to_string(), 600)]));

        let mut spec = basic_spec("over-budget");
        spec.qos = Some("budget".into());
        let job_id = submit_and_wait(&cm, spec);

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|j| j.job_id)
            .collect();
        assert!(
            !pending.contains(&job_id),
            "a QOS at its GrpWall budget must not be scheduled"
        );
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::QosGrpWallLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn grp_wall_budget_admits_while_consumption_is_below_the_cap() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        cm.qos_cache().insert(budget_qos(600));
        cm.grp_wall_cache()
            .seed(HashMap::from([("budget".to_string(), 599)]));

        let mut spec = basic_spec("within-budget");
        spec.qos = Some("budget".into());
        let job_id = submit_and_wait(&cm, spec);

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|j| j.job_id)
            .collect();
        assert!(
            pending.contains(&job_id),
            "consumption below the budget must still schedule"
        );
        // Differential: admitting alone would also hold with the limit deleted
        // outright. The reason must be absent too, which pins the boundary as
        // `>=` — 599 of 600 admits, and the sibling test above blocks at 600.
        assert_ne!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::QosGrpWallLimit,
            "a job inside its budget must not be tagged with the GrpWall reason"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn grp_wall_budget_is_unenforced_while_usage_is_unavailable() {
        // An unseeded cache is what accounting being disabled or unreachable looks
        // like. Scheduling must continue rather than stall on an unknown figure.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        cm.qos_cache().insert(budget_qos(1));

        let mut spec = basic_spec("no-usage-data");
        spec.qos = Some("budget".into());
        let job_id = submit_and_wait(&cm, spec);

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|j| j.job_id)
            .collect();
        assert!(
            pending.contains(&job_id),
            "an unavailable usage figure must not block scheduling"
        );
        assert_ne!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::QosGrpWallLimit,
            "an unknown figure must not be read as a breached budget"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_uses_qos_priority_as_base_when_user_omits_priority() {
        // When --priority is not set, the job's stored priority should equal
        // the QOS priority so it seeds the multiplicative formula correctly.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.qos_cache().insert(Qos {
            name: "burst".into(),
            priority: 100,
            ..Default::default()
        });
        cm.qos_cache().insert(Qos {
            name: "primus".into(),
            priority: 10000,
            ..Default::default()
        });

        let mut burst = basic_spec("burst");
        burst.qos = Some("burst".into());
        let burst_id = submit_and_wait(&cm, burst);

        let mut primus = basic_spec("primus");
        primus.qos = Some("primus".into());
        let primus_id = submit_and_wait(&cm, primus);

        let burst_job = cm.get_job(burst_id).unwrap();
        let primus_job = cm.get_job(primus_id).unwrap();
        // base = DEFAULT_PRIORITY.saturating_add_signed(qos.priority)
        assert_eq!(
            burst_job.priority,
            DEFAULT_PRIORITY.saturating_add_signed(100),
            "burst base must be DEFAULT_PRIORITY + qos.priority"
        );
        assert_eq!(
            primus_job.priority,
            DEFAULT_PRIORITY.saturating_add_signed(10000),
            "primus base must be DEFAULT_PRIORITY + qos.priority"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_explicit_priority_overrides_qos_priority() {
        // An explicit --priority flag must win over the QOS priority so users
        // can still pin a job's priority independent of its QOS tier.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.qos_cache().insert(Qos {
            name: "high".into(),
            priority: 9999,
            ..Default::default()
        });

        let mut spec = basic_spec("explicit-prio");
        spec.priority = Some(42);
        spec.qos = Some("high".into());
        let job_id = submit_and_wait(&cm, spec);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(
            job.priority, 42,
            "explicit --priority must not be overridden by qos.priority"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_job_suspend_retains_allocation() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "preempt-suspend", "worker1");
        let outcome = cm
            .preempt_job_with_provenance(
                job_id,
                PreemptMode::Suspend,
                Some(42),
                Some("highprio".into()),
            )
            .unwrap();
        assert_eq!(outcome, PreemptOutcome::Suspended);
        settle(&cm, job_id, JobState::Suspended);

        // Suspend retains the allocation (the process is only SIGSTOP'd).
        assert_eq!(cm.node_metrics().alloc_cpus, 2);

        // Provenance is persisted via the WAL op and survives replication.
        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.preempted_by, Some(42));
        assert_eq!(job.preempt_mode.as_deref(), Some("Suspend"));
        assert_eq!(job.preempt_qos.as_deref(), Some("highprio"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_job_off_mode_is_rejected() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "preempt-off", "worker1");
        assert!(cm
            .preempt_job_with_provenance(job_id, PreemptMode::Off, Some(99), None)
            .is_err());
        // Job keeps running; nothing was preempted.
        assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Running);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_job_rejects_non_running() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let job_id = submit_and_wait(&cm, basic_spec("still-pending"));
        assert!(cm
            .preempt_job_with_provenance(job_id, PreemptMode::Requeue, Some(99), None)
            .is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_state_change_idempotent_on_replay() {
        // WAL replay re-applies committed entries. A terminal job whose
        // completion entry is replayed must stay terminal without erroring or
        // re-running finalize side effects.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("replay")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));

        // Re-applying the same running transition is a NoOp, not an error.
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        assert_eq!(cm.get_job(1).unwrap().state, JobState::Running);

        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["worker1".into()],
            resources: alloc.clone(),
            per_node_alloc: per_node_for(&["worker1"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });
        cm.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: 0,
            state: JobState::Completed,
        });

        // Replaying the terminal complete: still Completed, resources still freed.
        let replayed = cm.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: 0,
            state: JobState::Completed,
        });
        assert!(replayed.jobs_finalized.is_empty());
        assert_eq!(cm.get_job(1).unwrap().state, JobState::Completed);
        assert_eq!(cm.get_node("worker1").unwrap().alloc_resources.cpus, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_requeue_state_change_not_double_counted_on_replay() {
        // A replayed Preempted->Pending entry must not double-increment
        // requeue_count or re-wipe allocation fields.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("requeue-replay")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobComplete {
            job_id: 1,
            exit_code: -1,
            state: JobState::Preempted,
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Preempted,
            JobState::Pending,
        ));
        assert_eq!(cm.get_job(1).unwrap().requeue_count, 1);

        // Replay the same requeue transition (job already Pending): NoOp.
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Preempted,
            JobState::Pending,
        ));
        assert_eq!(
            cm.get_job(1).unwrap().requeue_count,
            1,
            "replayed requeue must not double-count"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_preempt_requeue_is_atomic_and_replay_deterministic() {
        // A single JobPreemptRequeue op takes a RUNNING job to Pending-with-hold
        // AND frees its nodes AND finalizes the prior run as PREEMPTED for
        // accounting — no intermediate state. Replay applies the exact begin_time
        // and is a NoOp (no double-count, no drift, no re-dealloc).
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("preempt-replay")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));
        let alloc = scalar_alloc(2, 4000);
        cm.apply_operation(&WalOperation::JobStart {
            job_id: 1,
            nodes: vec!["worker1".into()],
            resources: alloc.clone(),
            per_node_alloc: per_node_for(&["worker1"], alloc),
            srun_step_dispatch: false,
            run_attempt: 0,
        });
        assert_eq!(cm.get_node("worker1").unwrap().alloc_resources.cpus, 2);

        let begin_time = Utc::now() + chrono::Duration::seconds(5);
        // Apply without provenance fields — simulates a WAL entry written by an
        // older controller; new fields default to None and must not panic.
        let resp = cm.apply_operation(&WalOperation::JobPreemptRequeue {
            job_id: 1,
            begin_time,
            preempted_by: None,
            preempt_qos: None,
        });
        // One op finalizes the prior run as PREEMPTED (drives accounting) ...
        assert_eq!(resp.jobs_finalized.len(), 1);
        assert_eq!(resp.jobs_finalized[0].state, JobState::Preempted);
        // ... and the job is Pending-with-hold with nodes freed.
        let job = cm.get_job(1).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.spec.begin_time, Some(begin_time));
        assert_eq!(job.pending_reason, PendingReason::Preempted);
        assert_eq!(job.preempted_by, None, "no preempted_by when field absent");
        assert_eq!(job.preempt_mode.as_deref(), Some("Requeue"));
        assert_eq!(job.preempt_requeue_count, 1);
        assert_eq!(job.requeue_count, 0);
        assert!(job.allocated_nodes.is_empty());
        assert_eq!(cm.get_node("worker1").unwrap().alloc_resources.cpus, 0);

        // Replay the identical entry: job is already Pending -> NoOp.
        let replay = cm.apply_operation(&WalOperation::JobPreemptRequeue {
            job_id: 1,
            begin_time,
            preempted_by: None,
            preempt_qos: None,
        });
        assert!(
            replay.jobs_finalized.is_empty(),
            "replayed preempt-requeue must not re-finalize"
        );
        let job = cm.get_job(1).unwrap();
        assert_eq!(
            job.spec.begin_time,
            Some(begin_time),
            "instant must not drift"
        );
        assert_eq!(
            job.preempt_requeue_count, 1,
            "replayed preempt-requeue must not double-count"
        );
        assert_eq!(cm.get_node("worker1").unwrap().alloc_resources.cpus, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_requeue_wal_apply_with_provenance_clears_on_restart() {
        // After a preempt-requeue the provenance fields are set; when the job is
        // re-dispatched (JobStart) they must be cleared so the new run's accounting
        // row does not inherit the previous preemption's data.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = run_job_on(&cm, "prov-clear", "worker1");
        cm.preempt_job_with_provenance(
            job_id,
            PreemptMode::Requeue,
            Some(77),
            Some("burst".into()),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Pending);

        let after_preempt = cm.get_job(job_id).unwrap();
        assert_eq!(after_preempt.preempted_by, Some(77));
        assert_eq!(after_preempt.preempt_mode.as_deref(), Some("Requeue"));

        // Expire the hold and re-dispatch.
        {
            let mut jobs = cm.jobs.write();
            if let Some(j) = jobs.get_mut(&job_id) {
                j.spec.begin_time = Some(Utc::now() - chrono::Duration::seconds(1));
            }
        }
        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        let after_start = cm.get_job(job_id).unwrap();
        assert_eq!(
            after_start.preempted_by, None,
            "provenance must be cleared on re-dispatch"
        );
        assert_eq!(
            after_start.preempt_mode, None,
            "provenance must be cleared on re-dispatch"
        );
        assert_eq!(
            after_start.preempt_qos, None,
            "provenance must be cleared on re-dispatch"
        );
    }

    /// Drive a job to RUNNING on `node` then finalize it as PREEMPTED via the
    /// apply path, returning its id. Mirrors the pre-fix WAL state that stranded
    /// jobs in PREEMPTED.
    fn preempted_job_on(cm: &ClusterManager, name: &str, node: &str) -> JobId {
        let job_id = run_job_on(cm, name, node);
        cm.apply_operation(&WalOperation::JobComplete {
            job_id,
            exit_code: -1,
            state: JobState::Preempted,
        });
        assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Preempted);
        job_id
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_node_complete_on_preempted_job_is_noop() {
        // A late/replayed node-completion for an already-PREEMPTED job must not
        // force an illegal PREEMPTED -> COMPLETED finalize: no state change, no
        // JobFinalized side-effect.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = preempted_job_on(&cm, "late-nodecomplete", "worker1");

        let resp = cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id,
            node_name: "worker1".into(),
            exit_code: 0,
            signal: 0,
        });
        assert!(
            resp.jobs_finalized.is_empty(),
            "stale node-complete must not re-finalize"
        );
        assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Preempted);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_job_complete_on_preempted_job_is_noop() {
        // Replaying a terminal JobComplete over an already-PREEMPTED job is a
        // silent no-op (this is the WAL-replay case for jobs 75/177).
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let job_id = preempted_job_on(&cm, "replay-complete", "worker1");

        let resp = cm.apply_operation(&WalOperation::JobComplete {
            job_id,
            exit_code: 0,
            state: JobState::Completed,
        });
        assert!(
            resp.jobs_finalized.is_empty(),
            "replayed complete over PREEMPTED must not re-finalize"
        );
        assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Preempted);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deadline_job_transitions_pending_to_deadline_with_deadline_reason() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let job_id = submit_and_wait(&cm, basic_spec("dl"));
        cm.deadline_job(job_id).unwrap();
        settle(&cm, job_id, JobState::Deadline);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Deadline);
        assert_eq!(job.pending_reason, PendingReason::DeadLine);
        assert_eq!(job.exit_code, Some(-1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deadline_job_rejects_non_pending_states() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 4, 8000);

        let job_id = submit_and_wait(&cm, basic_spec("running"));
        let resources = scalar_alloc(1, 1000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        assert!(cm.deadline_job(job_id).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deadline_reason_survives_update_pending_reasons() {
        // Regression guard for the field bug: scheduler_loop fires the
        // deadline path while update_pending_reasons is also running each
        // tick. If the guard in update_pending_reasons regresses, the reason
        // gets clobbered to NodeDown/Resources just before the WAL apply,
        // and the user sees the wrong cause in any audit log.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let job_id = submit_and_wait(&cm, basic_spec("dl-race"));

        // Manually mark DeadLine, then run update_pending_reasons over an
        // empty cluster_state (which would otherwise force Resources/NodeDown).
        {
            let mut jobs = cm.jobs.write();
            jobs.get_mut(&job_id).unwrap().pending_reason = PendingReason::DeadLine;
        }
        let empty_state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &[],
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        let snapshot = cm.get_job(job_id).unwrap();
        cm.update_pending_reasons(&[&snapshot], &empty_state);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.pending_reason, PendingReason::DeadLine);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fully_allocated_cluster_reports_resources_not_nodedown() {
        // Regression: a job waiting on a fully-busy cluster must report
        // Resources (matching Slurm), not NodeDown. An `Allocated` node is up,
        // just full; only genuine down/drain/error states are NodeDown.
        use spur_core::node::NodeState;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let job_id = submit_and_wait(&cm, basic_spec("busy"));
        let snapshot = cm.get_job(job_id).unwrap();

        // Fully-allocated (busy but UP) node -> Resources.
        let mut node = cm.get_node("n1").unwrap();
        node.state = NodeState::Allocated;
        node.alloc_resources = scalar_alloc(4, 8000);
        let nodes = vec![node];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        cm.update_pending_reasons(&[&snapshot], &state);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::Resources
        );

        // Genuinely down node -> NodeDown.
        let mut down = cm.get_node("n1").unwrap();
        down.state = NodeState::Down;
        let nodes = vec![down];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        cm.update_pending_reasons(&[&snapshot], &state);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::NodeDown
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn k8s_reserved_node_reports_k8s_reserved_not_resources() {
        // A job that would fit but for the node being claimed by k8s must report
        // K8sReserved, so `squeue` explains why an "idle"-looking node won't run it.
        use spur_core::k0s::K0sRole;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let job_id = submit_and_wait(&cm, basic_spec("wants-k8s-node"));
        let snapshot = cm.get_job(job_id).unwrap();

        // Idle node with capacity, but reserved for k8s -> K8sReserved.
        let mut node = cm.get_node("n1").unwrap();
        node.k0s_role = Some(K0sRole::Worker);
        let nodes = vec![node];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        cm.update_pending_reasons(&[&snapshot], &state);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::K8sReserved
        );

        // An exclusive job needs an idle node, so a busy k8s node would not run it
        // even if the role were cleared -> plain Resources, not K8sReserved.
        let mut busy = cm.get_node("n1").unwrap();
        busy.k0s_role = Some(K0sRole::Worker);
        busy.alloc_resources = scalar_alloc(2, 4000);
        busy.state = NodeState::Mixed;
        let nodes = vec![busy];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        let mut excl_spec = basic_spec("excl");
        excl_spec.exclusive = true;
        let excl = submit_and_wait(&cm, excl_spec);
        let excl_snap = cm.get_job(excl).unwrap();
        cm.update_pending_reasons(&[&excl_snap], &state);
        assert_eq!(
            cm.get_job(excl).unwrap().pending_reason,
            PendingReason::Resources
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nodelist_that_cannot_match_reports_req_node_not_avail() {
        // A job pinned to a node that isn't idle/usable must report
        // ReqNodeNotAvail, not Priority (as if merely queued behind others).
        use spur_core::node::NodeState;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        register_node(&cm, "n2", 4, 8000);

        let mut spec = basic_spec("pinned");
        spec.nodelist = Some("n1".into());
        let job_id = submit_and_wait(&cm, spec);
        let snapshot = cm.get_job(job_id).unwrap();

        // n1 is drained (not schedulable); n2 is idle but excluded by nodelist.
        let mut n1 = cm.get_node("n1").unwrap();
        n1.state = NodeState::Drain;
        let n2 = cm.get_node("n2").unwrap();
        let nodes = vec![n1, n2];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        cm.update_pending_reasons(&[&snapshot], &state);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::ReqNodeNotAvail
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn additive_nodelist_fill_shortage_reports_resources() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        register_node(&cm, "n2", 4, 8000);
        register_node(&cm, "n3", 4, 8000);

        let mut spec = basic_spec("fill-shortage");
        spec.num_nodes = 3;
        spec.num_tasks = 3;
        spec.nodelist = Some("n1".into());
        spec.exclude = Some("n3".into());
        let job_id = submit_and_wait(&cm, spec);
        let snapshot = cm.get_job(job_id).unwrap();

        let nodes = vec![
            cm.get_node("n1").unwrap(),
            cm.get_node("n2").unwrap(),
            cm.get_node("n3").unwrap(),
        ];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        cm.update_pending_reasons(&[&snapshot], &state);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::Resources
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn additive_nodelist_unavailable_listed_node_reports_req_node_not_avail() {
        use spur_core::node::NodeState;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        register_node(&cm, "n2", 4, 8000);
        register_node(&cm, "n3", 4, 8000);
        register_node(&cm, "n4", 4, 8000);

        let mut spec = basic_spec("listed-down");
        spec.num_nodes = 3;
        spec.num_tasks = 3;
        spec.nodelist = Some("n1".into());
        let job_id = submit_and_wait(&cm, spec);
        let snapshot = cm.get_job(job_id).unwrap();

        let mut n1 = cm.get_node("n1").unwrap();
        n1.state = NodeState::Drain;
        let nodes = vec![
            n1,
            cm.get_node("n2").unwrap(),
            cm.get_node("n3").unwrap(),
            cm.get_node("n4").unwrap(),
        ];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        cm.update_pending_reasons(&[&snapshot], &state);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::ReqNodeNotAvail
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn additive_nodelist_fill_blocked_by_maintenance_reports_reserved_maintenance() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        register_node(&cm, "n2", 4, 8000);
        register_node(&cm, "n3", 4, 8000);
        register_node(&cm, "n4", 4, 8000);

        let mut spec = basic_spec("fill-maintenance");
        spec.num_nodes = 3;
        spec.num_tasks = 3;
        spec.nodelist = Some("n1".into());
        let job_id = submit_and_wait(&cm, spec);
        let snapshot = cm.get_job(job_id).unwrap();

        let nodes = vec![
            cm.get_node("n1").unwrap(),
            cm.get_node("n2").unwrap(),
            cm.get_node("n3").unwrap(),
            cm.get_node("n4").unwrap(),
        ];
        let now = Utc::now();
        let reservations = vec![Reservation {
            name: "maintenance".into(),
            start_time: now + chrono::Duration::minutes(30),
            end_time: now + chrono::Duration::hours(2),
            nodes: vec!["n2".into(), "n3".into(), "n4".into()],
            accounts: Vec::new(),
            users: Vec::new(),
            flags: spur_core::reservation::ReservationFlags {
                maint: true,
                ..Default::default()
            },
            owner: String::new(),
        }];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &reservations,
            topology: None,
        };
        cm.update_pending_reasons(&[&snapshot], &state);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::ReservedMaintenance
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn additive_nodelist_listed_node_blocked_by_maintenance_reports_reserved_maintenance() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        register_node(&cm, "n2", 4, 8000);
        register_node(&cm, "n3", 4, 8000);
        register_node(&cm, "n4", 4, 8000);

        let mut spec = basic_spec("listed-maintenance");
        spec.num_nodes = 3;
        spec.num_tasks = 3;
        spec.nodelist = Some("n1".into());
        let job_id = submit_and_wait(&cm, spec);
        let snapshot = cm.get_job(job_id).unwrap();

        let nodes = vec![
            cm.get_node("n1").unwrap(),
            cm.get_node("n2").unwrap(),
            cm.get_node("n3").unwrap(),
            cm.get_node("n4").unwrap(),
        ];
        let now = Utc::now();
        let reservations = vec![Reservation {
            name: "maintenance".into(),
            start_time: now + chrono::Duration::minutes(30),
            end_time: now + chrono::Duration::hours(2),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: Vec::new(),
            flags: spur_core::reservation::ReservationFlags {
                maint: true,
                ..Default::default()
            },
            owner: String::new(),
        }];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &reservations,
            topology: None,
        };
        cm.update_pending_reasons(&[&snapshot], &state);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::ReservedMaintenance
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn more_nodes_than_exist_reports_partition_node_limit() {
        // B-05: a job needing more nodes than the partition physically has must
        // report PartitionNodeLimit (Slurm parity, verified on slurm 25.11.6),
        // not Priority — it can never be scheduled by waiting.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        register_node(&cm, "n2", 4, 8000);

        let mut spec = basic_spec("toobig");
        spec.num_nodes = 3; // only 2 nodes exist
        spec.num_tasks = 3;
        let job_id = submit_and_wait(&cm, spec);
        let snapshot = cm.get_job(job_id).unwrap();

        let nodes = vec![cm.get_node("n1").unwrap(), cm.get_node("n2").unwrap()];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        cm.update_pending_reasons(&[&snapshot], &state);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::PartitionNodeLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unmatchable_constraint_reports_bad_constraints() {
        // A --constraint no node carries can never schedule -> BadConstraints.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);

        let mut spec = basic_spec("feat");
        spec.constraint = Some("mi300x".into());
        let job_id = submit_and_wait(&cm, spec);
        let snapshot = cm.get_job(job_id).unwrap();

        // Node has no features.
        let nodes = vec![cm.get_node("n1").unwrap()];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        cm.update_pending_reasons(&[&snapshot], &state);
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::BadConstraints
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_partition_config_reason() {
        // A job exceeding partition max_nodes that reaches the scheduler (e.g. a
        // replayed pre-upgrade WAL entry predating submit-time bounds checking)
        // must be tagged PartitionConfig and dropped, not scheduled. Fresh
        // submits are rejected earlier (see submit_rejects_node_bounds_*).
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].max_nodes = Some(1);
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("toobig");
        spec.partition = Some("default".into());
        spec.num_nodes = 2;
        spec.num_tasks = 2;
        let job_id = 1;
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id,
            spec: Box::new(spec),
        });
        wait_for("job applied", || cm.get_job(job_id).is_some());

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::PartitionConfig
        );
        // pending_jobs() must agree: the job is dropped, not scheduled.
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "structurally-unschedulable job must be dropped from scheduling"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_time_limit_and_config_reasons() {
        // A max_time violation reports the specific PartitionTimeLimit reason;
        // a min_nodes violation reports the generic PartitionConfig reason.
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].max_time = Some("00:10:00".into()); // 10 min cap
        config.partitions[0].min_nodes = 2;
        let cm = test_cluster_with_config(&dir, config).await;

        // Meets min_nodes but exceeds max_time: with EnforcePartLimits off
        // (the default) the time cap is not a submit-time bound, so this is
        // admitted and pends with the specific PartitionTimeLimit reason.
        let mut over_time = basic_spec("overtime");
        over_time.partition = Some("default".into());
        over_time.num_nodes = 2;
        over_time.num_tasks = 2;
        over_time.time_limit = Some(chrono::Duration::hours(1));
        let t_id = submit_and_wait(&cm, over_time);

        // Below min_nodes is rejected at submit now, so inject directly to
        // exercise the scheduler's PartitionConfig tagging for replayed entries.
        let mut under_nodes = basic_spec("undernodes");
        under_nodes.partition = Some("default".into());
        under_nodes.num_nodes = 1; // below min_nodes=2
        let n_id = 999;
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: n_id,
            spec: Box::new(under_nodes),
        });
        wait_for("undernodes applied", || cm.get_job(n_id).is_some());

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(t_id).unwrap().pending_reason,
            PendingReason::PartitionTimeLimit,
            "time_limit over partition max_time -> PartitionTimeLimit"
        );
        assert_eq!(
            cm.get_job(n_id).unwrap().pending_reason,
            PendingReason::PartitionConfig,
            "num_nodes below partition min_nodes -> PartitionConfig"
        );
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == t_id),
            "time-blocked job must be dropped from scheduling"
        );
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == n_id),
            "min_nodes-blocked job must be dropped from scheduling"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_partition_inactive_when_not_up() {
        // Non-Up partition -> job admitted and held PENDING with PartitionInactive.
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].state = "DOWN".into();
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("downpart");
        spec.partition = Some("default".into());
        let job_id = submit_and_wait(&cm, spec);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::PartitionInactive
        );
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "job in a non-Up partition must be dropped from scheduling"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_begin_time_reason_for_a_deferred_job() {
        // A --begin deferral used to report None, the same string shown for a job
        // the scheduler simply had not reached yet, leaving no way to tell them
        // apart from squeue or scontrol.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        let mut spec = basic_spec("deferred");
        spec.begin_time = Some(Utc::now() + chrono::Duration::hours(1));
        let job_id = submit_and_wait(&cm, spec);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::BeginTime
        );
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "a deferred job must stay out of scheduling"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn begin_time_reason_gives_way_once_the_hold_lapses() {
        // A sticky BeginTime would keep claiming the job waits on a start time
        // that has already passed.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        let mut spec = basic_spec("begin-lapse");
        spec.begin_time = Some(Utc::now() + chrono::Duration::hours(1));
        let job_id = submit_and_wait(&cm, spec);
        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::BeginTime
        );

        {
            let mut jobs = cm.jobs.write();
            jobs.get_mut(&job_id).unwrap().spec.begin_time =
                Some(Utc::now() - chrono::Duration::seconds(1));
        }

        assert!(
            cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "an elapsed hold must return the job to scheduling"
        );

        // An empty cluster forces a real wait reason, which must win.
        let empty_state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &[],
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        let snapshot = cm.get_job(job_id).unwrap();
        cm.update_pending_reasons(&[&snapshot], &empty_state);
        assert_ne!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::BeginTime,
            "the real wait reason must replace a lapsed hold"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_structural_blocker_outranks_a_begin_time_hold() {
        // Both apply. The partition explains why the job cannot run at all,
        // which is more useful than naming a start time it will never reach.
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].state = "DOWN".into();
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("down-and-deferred");
        spec.partition = Some("default".into());
        spec.begin_time = Some(Utc::now() + chrono::Duration::hours(1));
        let job_id = submit_and_wait(&cm, spec);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::PartitionInactive
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn multi_partition_job_reaches_scheduling_when_one_partition_is_eligible() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].state = "DOWN".into();
        config.partitions[0].max_nodes = Some(1);
        let mut batch = config.partitions[0].clone();
        batch.name = "batch".into();
        batch.default = false;
        batch.state = "UP".into();
        batch.max_nodes = Some(2);
        config.partitions.push(batch);
        let cm = test_cluster_with_config(&dir, config).await;

        let mut spec = basic_spec("multi-partition");
        spec.partition = Some("default, batch".into());
        spec.num_nodes = 2;
        spec.num_tasks = 2;
        let job_id = submit_and_wait(&cm, spec);

        assert!(cm.pending_jobs().iter().any(|job| job.job_id == job_id));
        assert_eq!(
            cm.get_job(job_id).unwrap().spec.partition.as_deref(),
            Some("default, batch")
        );
    }

    #[test]
    fn partition_block_rejects_mixed_known_and_unknown_partitions() {
        let job = Job::new(
            1,
            JobSpec {
                partition: Some("default,missing".into()),
                ..basic_spec("mixed-partitions")
            },
        );
        let partitions = vec![Partition {
            name: "default".into(),
            ..Default::default()
        }];

        assert_eq!(
            partition_block(&job, &partitions),
            Some(PendingReason::PartitionConfig)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_account_not_in_allow_accounts() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].allow_accounts = vec!["research".into(), "faculty".into()];
        let cm = test_cluster_with_config(&dir, cfg).await;
        cm.association_cache()
            .insert_association("testuser", "student");

        let mut spec = basic_spec("badacct");
        spec.account = Some("student".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("account 'student' not allowed on partition 'default'")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_accepts_account_in_allow_accounts() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].allow_accounts = vec!["research".into(), "faculty".into()];
        let cm = test_cluster_with_config(&dir, cfg).await;
        cm.association_cache()
            .insert_association("testuser", "research");

        let mut spec = basic_spec("goodacct");
        spec.account = Some("research".into());
        assert!(cm.submit_job(spec).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_deny_accounts() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].deny_accounts = vec!["student".into()];
        let cm = test_cluster_with_config(&dir, cfg).await;
        cm.association_cache()
            .insert_association("testuser", "student");

        let mut spec = basic_spec("denied");
        spec.account = Some("student".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("account 'student' denied on partition 'default'")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_allow_accounts_uses_default_account_when_unset() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].allow_accounts = vec!["research".into()];
        let cm = test_cluster_with_config(&dir, cfg).await;
        cm.association_cache()
            .insert_default_account("testuser", "research");

        let spec = basic_spec("defaultacct");
        assert!(cm.submit_job(spec).is_ok());
    }

    fn qos_with_limits(name: &str, limits: spur_core::accounting::QosLimits, deny: bool) -> Qos {
        Qos {
            name: name.into(),
            limits,
            deny_on_limit: deny,
            ..Default::default()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_denies_assoc_max_submit_jobs_zero_blocks_all() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.association_cache().insert_limits(
            "testuser",
            "research",
            AccountLimits {
                max_submit_jobs: Some(0),
                ..Default::default()
            },
        );
        let mut spec = basic_spec("blocked");
        spec.account = Some("research".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert_submit_denied(&err, "AssocMaxSubmitJobLimit");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_denies_assoc_max_submit_jobs_over_count() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.association_cache().insert_limits(
            "testuser",
            "research",
            AccountLimits {
                max_submit_jobs: Some(1),
                ..Default::default()
            },
        );
        let mut first = basic_spec("first");
        first.account = Some("research".into());
        assert!(cm.submit_job(first).is_ok());

        let mut second = basic_spec("second");
        second.account = Some("research".into());
        let err = cm.submit_job(second).unwrap_err();
        assert_submit_denied(&err, "AssocMaxSubmitJobLimit");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_denies_assoc_wall_unconditionally() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.association_cache().insert_limits(
            "testuser",
            "research",
            AccountLimits {
                max_wall_minutes: Some(60),
                ..Default::default()
            },
        );
        let mut spec = basic_spec("longwall");
        spec.account = Some("research".into());
        spec.time_limit = Some(chrono::Duration::hours(2));
        let err = cm.submit_job(spec).unwrap_err();
        assert_submit_denied(&err, "AssocMaxWallDurationPerJobLimit");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_denies_array_that_overflows_submit_cap() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.association_cache().insert_limits(
            "testuser",
            "research",
            AccountLimits {
                max_submit_jobs: Some(5),
                ..Default::default()
            },
        );
        let mut spec = basic_spec("bigarray");
        spec.account = Some("research".into());
        spec.array_spec = Some("0-9".into()); // 10 tasks > 5
        let err = cm.submit_job(spec).unwrap_err();
        assert_submit_denied(&err, "AssocMaxSubmitJobLimit");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_gives_a_job_that_asked_for_no_wall_time_the_qos_max_wall() {
        // Without this the job runs unbounded and shows UNLIMITED, since the
        // time-limit watchdog has no deadline to enforce.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.association_cache()
            .insert_allowed_qos("testuser", "research", &["burst"]);
        cm.qos_cache().insert(qos_with_limits(
            "burst",
            spur_core::accounting::QosLimits {
                max_wall_minutes: Some(720),
                ..Default::default()
            },
            false,
        ));

        let mut spec = basic_spec("nolimit");
        spec.account = Some("research".into());
        spec.qos = Some("burst".into());
        spec.time_limit = None;

        let id = submit_and_wait(&cm, spec);
        assert_eq!(
            cm.get_job(id).unwrap().spec.time_limit,
            Some(chrono::Duration::minutes(720))
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_denies_qos_grp_submit_jobs() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.association_cache()
            .insert_allowed_qos("testuser", "research", &["normal"]);
        cm.qos_cache().insert(qos_with_limits(
            "normal",
            spur_core::accounting::QosLimits {
                grp_submit_jobs: Some(0),
                ..Default::default()
            },
            false,
        ));
        let mut spec = basic_spec("qosgrp");
        spec.account = Some("research".into());
        spec.qos = Some("normal".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert_submit_denied(&err, "QOSGrpSubmitJobsLimit");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_denies_qos_max_submit_per_account() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.association_cache()
            .insert_allowed_qos("testuser", "research", &["normal"]);
        cm.qos_cache().insert(qos_with_limits(
            "normal",
            spur_core::accounting::QosLimits {
                max_submit_jobs_per_account: Some(0),
                ..Default::default()
            },
            false,
        ));
        let mut spec = basic_spec("qospa");
        spec.account = Some("research".into());
        spec.qos = Some("normal".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert_submit_denied(&err, "MaxSubmitJobsPerAccount");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_denies_qos_standalone_tres_only_with_deny_on_limit() {
        let mut cap = TresRecord::new();
        cap.set(TresType::Cpu, 1);
        let limits = spur_core::accounting::QosLimits {
            max_tres_per_job: Some(cap),
            ..Default::default()
        };

        // Without DenyOnLimit the unsatisfiable request is admitted (pends).
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.association_cache()
            .insert_allowed_qos("testuser", "research", &["normal"]);
        cm.qos_cache()
            .insert(qos_with_limits("normal", limits.clone(), false));
        let mut spec = basic_spec("pends");
        spec.account = Some("research".into());
        spec.qos = Some("normal".into());
        spec.num_tasks = 2; // 2 CPUs > cap of 1
        assert!(cm.submit_job(spec).is_ok());

        // With DenyOnLimit the same request is denied at submit.
        let dir2 = TempDir::new().unwrap();
        let cm2 = test_cluster(&dir2).await;
        cm2.association_cache()
            .insert_allowed_qos("testuser", "research", &["normal"]);
        cm2.qos_cache()
            .insert(qos_with_limits("normal", limits, true));
        let mut spec2 = basic_spec("denied");
        spec2.account = Some("research".into());
        spec2.qos = Some("normal".into());
        spec2.num_tasks = 2;
        let err = cm2.submit_job(spec2).unwrap_err();
        assert_submit_denied(&err, "QOSMaxCpuPerJobLimit");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_admits_over_maxtime_when_enforcement_off() {
        // Default EnforcePartLimits=No: an over-MaxTime job is admitted (and
        // will pend with PartitionTimeLimit), not rejected at submit.
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].max_time = Some("00:10:00".into());
        let cm = test_cluster_with_config(&dir, cfg).await;

        let mut spec = basic_spec("overtime");
        spec.time_limit = Some(chrono::Duration::hours(1));
        assert!(cm.submit_job(spec).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_over_maxtime_when_enforcement_all() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].max_time = Some("00:10:00".into());
        cfg.scheduler.enforce_part_limits = EnforcePartLimits::All;
        let cm = test_cluster_with_config(&dir, cfg).await;

        let mut spec = basic_spec("overtime");
        spec.time_limit = Some(chrono::Duration::hours(1));
        let err = cm.submit_job(spec).unwrap_err();
        let SubmitError::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(
            msg.contains("Requested time limit is invalid"),
            "expected Slurm-style message, got: {msg}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_accepts_within_maxtime_when_enforcement_all() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].max_time = Some("01:00:00".into());
        cfg.scheduler.enforce_part_limits = EnforcePartLimits::All;
        let cm = test_cluster_with_config(&dir, cfg).await;

        let mut spec = basic_spec("withintime");
        spec.time_limit = Some(chrono::Duration::minutes(30));
        assert!(cm.submit_job(spec).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_unlimited_partition_accepts_any_time_under_enforcement() {
        // UNLIMITED MaxTime imposes no cap even with enforcement on (Slurm parity).
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].max_time = Some("UNLIMITED".into());
        cfg.scheduler.enforce_part_limits = EnforcePartLimits::All;
        // Disable the cluster fallback so the huge request survives to submit.
        cfg.scheduler.default_time_limit_minutes = 0;
        let cm = test_cluster_with_config(&dir, cfg).await;

        let mut spec = basic_spec("yearlong");
        spec.time_limit = Some(chrono::Duration::days(365));
        assert!(cm.submit_job(spec).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_enforcement_any_rejects_when_no_partition_fits() {
        // ANY rejects only when the request fits none of the requested
        // partitions; the error names the offending partition.
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].max_time = Some("00:10:00".into());
        let mut other = cfg.partitions[0].clone();
        other.name = "other".into();
        other.default = false;
        other.max_time = Some("00:20:00".into());
        cfg.partitions.push(other);
        cfg.scheduler.enforce_part_limits = EnforcePartLimits::Any;
        let cm = test_cluster_with_config(&dir, cfg).await;

        let mut spec = basic_spec("toolong");
        spec.partition = Some("default,other".into());
        spec.time_limit = Some(chrono::Duration::hours(1));
        let err = cm.submit_job(spec).unwrap_err();
        let SubmitError::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(
            msg.contains("Requested time limit is invalid") && msg.contains("partition"),
            "error must name the offending partition: {msg}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_enforcement_any_accepts_if_one_partition_fits() {
        // Two partitions: one 10-min cap, one 2-hour cap. A 1-hour job fits the
        // second, so ANY admits it; ALL would reject it.
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].max_time = Some("00:10:00".into());
        let mut roomy = cfg.partitions[0].clone();
        roomy.name = "roomy".into();
        roomy.default = false;
        roomy.max_time = Some("02:00:00".into());
        cfg.partitions.push(roomy);
        cfg.scheduler.enforce_part_limits = EnforcePartLimits::Any;
        let cm = test_cluster_with_config(&dir, cfg).await;

        let mut spec = basic_spec("multi");
        spec.partition = Some("default,roomy".into());
        spec.time_limit = Some(chrono::Duration::hours(1));
        assert!(cm.submit_job(spec).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_multi_partition_all_admits_untimed_job() {
        // Regression: a `-t`-less job to two partitions under ALL must not be
        // rejected by a wall-time we auto-assign. The default is derived from the
        // smaller MaxTime so it fits both, and an auto-default is never enforced.
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].max_time = Some("02:00:00".into());
        let mut small = cfg.partitions[0].clone();
        small.name = "small".into();
        small.default = false;
        small.max_time = Some("01:00:00".into());
        cfg.partitions.push(small);
        cfg.scheduler.enforce_part_limits = EnforcePartLimits::All;
        cfg.scheduler.default_time_limit_minutes = 90;
        let cm = test_cluster_with_config(&dir, cfg).await;

        let mut spec = basic_spec("untimed");
        spec.partition = Some("default,small".into());
        spec.time_limit = None;
        assert!(cm.submit_job(spec).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_negative_time_limit() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster_with_config(&dir, test_config()).await;

        let mut spec = basic_spec("neg");
        spec.time_limit = Some(chrono::Duration::minutes(-5));
        let err = cm.submit_job(spec).unwrap_err();
        let SubmitError::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(msg.contains("negative"), "got: {msg}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_empty_partition_string_applies_default_and_enforces_allow_accounts() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].allow_accounts = vec!["research".into()];
        let cm = test_cluster_with_config(&dir, cfg).await;
        cm.association_cache()
            .insert_association("testuser", "student");

        let mut spec = basic_spec("emptypart");
        spec.partition = Some(String::new());
        spec.account = Some("student".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("account 'student' not allowed on partition 'default'")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_allow_accounts_when_account_unresolved() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].allow_accounts = vec!["research".into()];
        let cm = test_cluster_with_config(&dir, cfg).await;
        cm.association_cache().set_loaded_without_associations();

        let spec = basic_spec("noacct");
        let err = cm.submit_job(spec).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("no account for user 'testuser' on partition 'default'")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_deny_accounts_with_default_account() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].deny_accounts = vec!["student".into()];
        let cm = test_cluster_with_config(&dir, cfg).await;
        cm.association_cache()
            .insert_default_account("testuser", "student");

        let spec = basic_spec("denied-default");
        let err = cm.submit_job(spec).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("account 'student' denied on partition 'default'")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_spoofed_account_not_in_membership() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].allow_accounts = vec!["research".into()];
        let cm = test_cluster_with_config(&dir, cfg).await;
        cm.association_cache()
            .insert_association("testuser", "student-z");
        cm.association_cache()
            .insert_association("testuser", "student-a");

        let mut spec = basic_spec("spoof");
        spec.account = Some("research".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid(
                "user 'testuser' is not associated with account 'research'. Accounts associated with this user: [student-a, student-z]."
            )
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_account_when_user_has_no_associations() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.association_cache().set_loaded_without_associations();

        let mut spec = basic_spec("unassociated");
        spec.account = Some("research".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid(
                "user 'testuser' has no account associations. Contact your cluster admin to run: sacctmgr add user name=testuser account=<account>"
            )
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_allow_accounts_ignores_deny_when_both_list_account() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].allow_accounts = vec!["research".into()];
        cfg.partitions[0].deny_accounts = vec!["research".into()];
        let cm = test_cluster_with_config(&dir, cfg).await;
        cm.association_cache()
            .insert_association("testuser", "research");

        let mut spec = basic_spec("overlap");
        spec.account = Some("research".into());
        assert!(cm.submit_job(spec).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_enforces_allow_accounts_without_association_cache() {
        // Partition ACL is pure string matching — it must fire even when the
        // accounting association cache is empty (no Postgres backend running).
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions[0].allow_accounts = vec!["research".into()];
        let cm = test_cluster_with_config(&dir, cfg).await;

        let mut spec = basic_spec("nocache");
        spec.account = Some("student".into());
        assert!(
            cm.submit_job(spec).is_err(),
            "unlisted account must be rejected"
        );
    }

    /// An `AccountingConfig` whose `enabled()` is `true`/`false` (accounting is enabled iff a
    /// database URL is set), for the fail-closed-when-enabled account checks.
    fn acct_cfg_enabled(enabled: bool) -> spur_core::config::AccountingConfig {
        spur_core::config::AccountingConfig {
            database_url: if enabled {
                "postgresql://test".into()
            } else {
                String::new()
            },
            ..Default::default()
        }
    }

    #[test]
    fn validate_user_account_fails_closed_on_cold_cache_when_accounting_enabled() {
        // The inducible cache-cold window (fresh start / restart / DB unreachable at boot) must not
        // wave an account-scoped submission through — that would clear every partition ACL at once.
        let cache = AssociationCache::new(); // never loaded
        let mut spec = basic_spec("cold");
        spec.account = Some("tenant-b".into());
        let err = validate_user_account(&spec, &cache, &acct_cfg_enabled(true)).unwrap_err();
        assert!(
            matches!(&err, SubmitError::Unavailable(m) if m.contains("temporarily unavailable")),
            "cold cache with accounting enabled must fail closed (retryable), got {err:?}"
        );
    }

    #[test]
    fn validate_user_account_permits_cold_cache_when_accounting_disabled() {
        // With accounting off there is no association database, so an account is an unverified free
        // label and there is no fence to hold — stay permissive (matches prior behaviour).
        let cache = AssociationCache::new();
        let mut spec = basic_spec("cold-noacct");
        spec.account = Some("tenant-b".into());
        assert!(validate_user_account(&spec, &cache, &acct_cfg_enabled(false)).is_ok());
    }

    #[test]
    fn validate_user_account_allows_real_member_on_loaded_cache() {
        let cache = AssociationCache::new();
        cache.insert_association("testuser", "research");
        let mut spec = basic_spec("member");
        spec.account = Some("research".into());
        assert!(validate_user_account(&spec, &cache, &acct_cfg_enabled(true)).is_ok());
    }

    #[test]
    fn validate_user_account_rejects_nonmember_on_loaded_cache() {
        let cache = AssociationCache::new();
        cache.insert_association("testuser", "research");
        let mut spec = basic_spec("spoof");
        spec.account = Some("tenant-b".into());
        let err = validate_user_account(&spec, &cache, &acct_cfg_enabled(true)).unwrap_err();
        assert!(
            matches!(&err, SubmitError::InvalidArgument(m) if m.contains("not associated with account 'tenant-b'")),
            "got {err:?}"
        );
    }

    #[test]
    fn validate_user_account_ignores_an_unset_account() {
        // No account requested — nothing to verify — even accounting-enabled + cold cache is fine.
        let cache = AssociationCache::new();
        let spec = basic_spec("noacct");
        assert!(validate_user_account(&spec, &cache, &acct_cfg_enabled(true)).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_fails_closed_on_cold_cache_when_accounting_enabled() {
        // End-to-end: accounting configured (a DB URL is set) but the cache has not loaded — an
        // account-scoped submit is denied rather than admitted across the cold-cache window.
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.accounting.database_url = "postgres://unused-in-test".into();
        let cm = test_cluster_with_config(&dir, cfg).await;
        // Deliberately do NOT load the association cache (mirrors a controller that never completed
        // its first fetch).
        let mut spec = basic_spec("coldsubmit");
        spec.account = Some("tenant-b".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert!(
            matches!(&err, SubmitError::Unavailable(m) if m.contains("temporarily unavailable")),
            "cold cache must fail closed end-to-end (retryable), got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_still_rejects_nonexistent_partition() {
        // Unknown partition must still be rejected at submit, not held pending.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let mut spec = basic_spec("badpart");
        spec.partition = Some("does-not-exist".into());
        assert!(
            cm.submit_job(spec).is_err(),
            "submitting to an unknown partition must error"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_rejects_nonexistent_partition_in_or_list() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let mut spec = basic_spec("badpart");
        spec.partition = Some("default, does-not-exist".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("partition 'does-not-exist' not found")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_enforces_account_access_on_every_requested_partition() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].allow_accounts = vec!["research".into()];
        let mut restricted = config.partitions[0].clone();
        restricted.name = "restricted".into();
        restricted.default = false;
        restricted.allow_accounts = vec!["faculty".into()];
        config.partitions.push(restricted);
        let cm = test_cluster_with_config(&dir, config).await;
        cm.association_cache()
            .insert_association("testuser", "research");

        let mut spec = basic_spec("mixed-access");
        spec.partition = Some("default,restricted".into());
        spec.account = Some("research".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("account 'research' not allowed on partition 'restricted'")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_partition_not_found_returns_invalid_argument() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let mut spec = basic_spec("badpart");
        spec.partition = Some("does-not-exist".into());
        let err = cm.submit_job(spec).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("partition 'does-not-exist' not found")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_reservation_reason() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let mut spec = basic_spec("resv");
        spec.reservation = Some("does-not-exist".into());
        let job_id = submit_and_wait(&cm, spec);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::Reservation
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reservation_overlap_rejected_without_flag() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        let base = Reservation {
            name: String::new(),
            start_time: now,
            end_time: now + chrono::Duration::hours(2),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: vec!["alice".into()],
            flags: Default::default(),
            owner: String::new(),
        };
        let mut r1 = base.clone();
        r1.name = "r1".into();
        cm.create_reservation(r1).unwrap();
        let mut r2 = base;
        r2.name = "r2".into();
        r2.start_time = now + chrono::Duration::hours(1);
        assert!(cm.create_reservation(r2).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reservation_overlap_allowed_with_overlap_flag() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        let base = Reservation {
            name: String::new(),
            start_time: now,
            end_time: now + chrono::Duration::hours(2),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: vec!["alice".into()],
            flags: Default::default(),
            owner: String::new(),
        };
        let mut r1 = base.clone();
        r1.name = "r1".into();
        cm.create_reservation(r1).unwrap();
        let mut r2 = base;
        r2.name = "r2".into();
        r2.start_time = now + chrono::Duration::hours(1);
        r2.flags.overlap = true;
        assert!(cm.create_reservation(r2).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_prioritize_active_reservation_targets() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        cm.create_reservation(Reservation {
            name: "r1".into(),
            start_time: now - chrono::Duration::minutes(5),
            end_time: now + chrono::Duration::hours(2),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: vec!["testuser".into()],
            flags: Default::default(),
            owner: String::new(),
        })
        .unwrap();

        let mut plain = basic_spec("plain");
        plain.priority = Some(5000);
        let plain_id = submit_and_wait(&cm, plain);

        let mut resv = basic_spec("resv");
        resv.priority = Some(1000);
        resv.reservation = Some("r1".into());
        let resv_id = submit_and_wait(&cm, resv);

        let pending = cm.pending_jobs();
        assert_eq!(pending.first().map(|j| j.job_id), Some(resv_id));
        assert!(pending.iter().any(|j| j.job_id == plain_id));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_applies_qos_priority_adjustment() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.qos_cache().insert(Qos {
            name: "low".into(),
            priority: -500,
            ..Default::default()
        });
        cm.qos_cache().insert(Qos {
            name: "high".into(),
            priority: 5000,
            ..Default::default()
        });

        // No explicit --priority: base = DEFAULT_PRIORITY.saturating_add_signed(qos.priority).
        // low (-500): base = 500, below DEFAULT_PRIORITY — penalty semantics preserved.
        // high (5000): base = 6000, above DEFAULT_PRIORITY — boost semantics preserved.
        let mut low = basic_spec("low");
        low.qos = Some("low".into());
        let low_id = submit_and_wait(&cm, low);

        let mut high = basic_spec("high");
        high.qos = Some("high".into());
        let high_id = submit_and_wait(&cm, high);

        let pending = cm.pending_jobs();
        let low_priority = pending
            .iter()
            .find(|j| j.job_id == low_id)
            .unwrap()
            .priority;
        let high_priority = pending
            .iter()
            .find(|j| j.job_id == high_id)
            .unwrap()
            .priority;
        assert!(
            high_priority > low_priority,
            "high-QoS job ({high_priority}) should outrank low-QoS job ({low_priority})"
        );
        assert!(
            low_priority < DEFAULT_PRIORITY,
            "penalty QoS job ({low_priority}) must rank below DEFAULT_PRIORITY ({DEFAULT_PRIORITY})"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_qos_ordering_holds_with_nonneutral_fairshare_and_age() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.qos_cache().insert(Qos {
            name: "low".into(),
            priority: 100,
            ..Default::default()
        });
        cm.qos_cache().insert(Qos {
            name: "high".into(),
            priority: 10000,
            ..Default::default()
        });

        // Even with high fairshare and near-max age on the low-QoS job, the
        // high-QoS base (11000) must outrank low-QoS base (1100) × 3.0 × 1.857.
        // low:  1100 * 3.0 * 1.857 * tier ≈ 6128
        // high: 11000 * 1.0 * 1.0 * tier = 11000
        cm.fairshare_cache().set_for_test("low-user", "", 3.0);

        let mut low = basic_spec("low");
        low.user = "low-user".into();
        low.qos = Some("low".into());
        let low_id = submit_and_wait(&cm, low);
        {
            let mut jobs = cm.jobs.write();
            jobs.get_mut(&low_id).unwrap().submit_time = Utc::now() - chrono::Duration::days(6);
        }

        let mut high = basic_spec("high");
        high.qos = Some("high".into());
        let high_id = submit_and_wait(&cm, high);

        let pending = cm.pending_jobs();
        let low_priority = pending
            .iter()
            .find(|j| j.job_id == low_id)
            .unwrap()
            .priority;
        let high_priority = pending
            .iter()
            .find(|j| j.job_id == high_id)
            .unwrap()
            .priority;
        assert!(
            high_priority > low_priority,
            "high-QoS job ({high_priority}) should still outrank low-QoS job ({low_priority}) \
             even with high fairshare and near-max age on the low-QoS job"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn current_effective_priority_multi_partition_uses_highest_priority_tier() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        {
            let mut partitions = cm.partitions.write();
            partitions.push(Partition {
                name: "low".into(),
                priority_tier: 1,
                ..Default::default()
            });
            partitions.push(Partition {
                name: "high".into(),
                priority_tier: 9,
                ..Default::default()
            });
        }

        // Built directly (not submitted) to isolate priority resolution from
        // unrelated eligibility filters like partition_block().
        let job = Job::new(
            1,
            JobSpec {
                partition: Some("low,high".into()),
                priority: Some(1000),
                ..basic_spec("multi")
            },
        );

        let priority = cm.current_effective_priority(&job, &cm.get_partitions());
        assert_eq!(
            priority, 9000,
            "multi-partition job should use the highest matched priority_tier (9), \
             not fall back to 1 because \"low,high\" isn't an exact partition name"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn resv_overrun_grace_delays_cancel() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.scheduler.resv_overrun_minutes = 30;
        let cm = test_cluster_with_config(&dir, config).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        cm.create_reservation(Reservation {
            name: "r1".into(),
            start_time: now - chrono::Duration::hours(1),
            end_time: now - chrono::Duration::minutes(5),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: vec!["testuser".into()],
            flags: Default::default(),
            owner: String::new(),
        })
        .unwrap();

        let mut spec = basic_spec("resv-run");
        spec.reservation = Some("r1".into());
        let job_id = submit_and_wait(&cm, spec);
        let res = scalar_alloc(1, 1000);
        cm.start_job(
            job_id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        cm.enforce_reservation_end_times();
        assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Running);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_skips_jobs_in_active_reservation_at_same_tier() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].preempt_mode = "cancel".into();
        let cm = test_cluster_with_config(&dir, config).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        cm.create_reservation(Reservation {
            name: "r1".into(),
            start_time: now - chrono::Duration::minutes(5),
            end_time: now + chrono::Duration::hours(2),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: vec!["testuser".into()],
            flags: Default::default(),
            owner: String::new(),
        })
        .unwrap();

        let mut low = basic_spec("low");
        low.priority = Some(100);
        low.reservation = Some("r1".into());
        let low_id = submit_and_wait(&cm, low);
        let res = scalar_alloc(1, 1000);
        cm.start_job(
            low_id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, low_id, JobState::Running);

        let mut high = basic_spec("high");
        high.priority = Some(10_000);
        let high_id = submit_and_wait(&cm, high);
        let high_job = cm.get_job(high_id).unwrap();
        let partitions = cm.get_partitions();

        crate::scheduler_loop::try_preempt(&cm, &partitions, &[&high_job], &cm.config().scheduler)
            .await;
        assert_eq!(cm.get_job(low_id).unwrap().state, JobState::Running);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_skips_jobs_on_unrelated_nodes() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].preempt_mode = "cancel".into();
        let cm = test_cluster_with_config(&dir, config).await;
        register_node(&cm, "n1", 8, 16000);
        register_node(&cm, "n2", 8, 16000);

        let mut low = basic_spec("low");
        low.priority = Some(100);
        low.nodelist = Some("n2".into());
        let low_id = submit_and_wait(&cm, low);
        let res = scalar_alloc(1, 1000);
        cm.start_job(
            low_id,
            vec!["n2".into()],
            res.clone(),
            per_node_for(&["n2"], res),
        )
        .unwrap();
        settle(&cm, low_id, JobState::Running);

        let mut high = basic_spec("high");
        high.priority = Some(10_000);
        high.nodelist = Some("n1".into());
        let high_id = submit_and_wait(&cm, high);
        let high_job = cm.get_job(high_id).unwrap();
        let partitions = cm.get_partitions();

        crate::scheduler_loop::try_preempt(&cm, &partitions, &[&high_job], &cm.config().scheduler)
            .await;
        assert_eq!(cm.get_job(low_id).unwrap().state, JobState::Running);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_triggers_when_qos_priority_differentiates_pending_job() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].preempt_mode = "cancel".into();
        let cm = test_cluster_with_config(&dir, config).await;
        register_node(&cm, "n1", 8, 16000);

        cm.qos_cache().insert(Qos {
            name: "high".into(),
            priority: 5000,
            ..Default::default()
        });

        // low job has no QOS (base=DEFAULT_PRIORITY=1000); high job's QOS
        // priority (5000) becomes its base, making it >2x the low job's
        // effective priority and crossing the preemption threshold.
        let mut low = basic_spec("low");
        low.priority = Some(1000);
        let low_id = submit_and_wait(&cm, low);
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            low_id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, low_id, JobState::Running);

        let mut high = basic_spec("high");
        high.qos = Some("high".into());
        submit_and_wait(&cm, high);

        let pending = cm.pending_jobs();
        let pending_refs: Vec<&Job> = pending.iter().collect();
        let partitions = cm.get_partitions();
        crate::scheduler_loop::try_preempt(&cm, &partitions, &pending_refs, &cm.config().scheduler)
            .await;

        settle(&cm, low_id, JobState::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn high_qos_job_preempts_low_qos_job_end_to_end() {
        // Full end-to-end: burst job runs, primus job arrives pending.
        // Neither sets --priority; both get their base from QOS.
        // burst (100) must be evicted to make room for primus (10000).
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].preempt_mode = "cancel".into();
        let cm = test_cluster_with_config(&dir, config).await;
        register_node(&cm, "n1", 8, 16000);

        cm.qos_cache().insert(Qos {
            name: "burst".into(),
            priority: 100,
            ..Default::default()
        });
        cm.qos_cache().insert(Qos {
            name: "primus".into(),
            priority: 10000,
            ..Default::default()
        });

        let mut burst = basic_spec("burst");
        burst.qos = Some("burst".into());
        let burst_id = submit_and_wait(&cm, burst);
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            burst_id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, burst_id, JobState::Running);

        let mut primus = basic_spec("primus");
        primus.qos = Some("primus".into());
        submit_and_wait(&cm, primus);

        // Confirm that primus effective > 2x burst effective (preemption threshold).
        let pending = cm.pending_jobs();
        let pending_refs: Vec<&Job> = pending.iter().collect();
        let partitions = cm.get_partitions();

        let burst_job = cm.get_job(burst_id).unwrap();
        let burst_effective = cm.current_effective_priority(&burst_job, &partitions);
        let primus_effective = pending_refs[0].priority;
        assert!(
            primus_effective >= burst_effective * 2,
            "primus effective ({primus_effective}) must be ≥2x burst ({burst_effective}) \
             for preemption to fire"
        );

        crate::scheduler_loop::try_preempt(&cm, &partitions, &pending_refs, &cm.config().scheduler)
            .await;

        settle(&cm, burst_id, JobState::Cancelled);
        assert_eq!(
            cm.node_metrics().alloc_cpus,
            0,
            "burst job's allocation must be freed after preemption"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_priority_disables_qos_based_preemption() {
        // When --priority is set explicitly on both jobs, QOS priority is not
        // applied as the base seed. Both jobs land at the same effective
        // priority and preemption does not fire — this is intentional: explicit
        // --priority opts the job out of QOS-based scheduling.
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].preempt_mode = "cancel".into();
        let cm = test_cluster_with_config(&dir, config).await;
        register_node(&cm, "n1", 8, 16000);

        cm.qos_cache().insert(Qos {
            name: "burst".into(),
            priority: 100,
            ..Default::default()
        });
        cm.qos_cache().insert(Qos {
            name: "primus".into(),
            priority: 10000,
            ..Default::default()
        });

        let mut burst = basic_spec("burst");
        burst.priority = Some(1000);
        burst.qos = Some("burst".into());
        let burst_id = submit_and_wait(&cm, burst);
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            burst_id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, burst_id, JobState::Running);

        let mut primus = basic_spec("primus");
        primus.priority = Some(1000);
        primus.qos = Some("primus".into());
        submit_and_wait(&cm, primus);

        let pending = cm.pending_jobs();
        let pending_refs: Vec<&Job> = pending.iter().collect();
        let partitions = cm.get_partitions();
        crate::scheduler_loop::try_preempt(&cm, &partitions, &pending_refs, &cm.config().scheduler)
            .await;

        // burst job must still be running — equal explicit priorities, no preemption.
        let burst_job = cm.get_job(burst_id).unwrap();
        assert_eq!(
            burst_job.state,
            JobState::Running,
            "burst job must keep running when both jobs have identical explicit --priority"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qos_preempt_hierarchy_blocks_preemption_when_not_in_allow_list() {
        // With preempt_type = qos_priority, a pending job may only preempt a
        // running job when the pending job's QOS lists the running QOS in its
        // `preempt` allow-list. Here "high" does NOT list "low", so even though
        // "high" has a priority gap > 2x, preemption must be blocked.
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].preempt_mode = "cancel".into();
        config.scheduler.preempt_type = spur_core::partition::PreemptType::QosPriority;
        let cm = test_cluster_with_config(&dir, config).await;
        register_node(&cm, "n1", 8, 16000);

        cm.qos_cache().insert(Qos {
            name: "low".into(),
            priority: 100,
            ..Default::default()
        });
        cm.qos_cache().insert(Qos {
            name: "high".into(),
            priority: 10000,
            preempt: vec![], // empty = not allowed to preempt anything
            ..Default::default()
        });

        let mut low = basic_spec("low");
        low.qos = Some("low".into());
        let low_id = submit_and_wait(&cm, low);
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            low_id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, low_id, JobState::Running);

        let mut high = basic_spec("high");
        high.qos = Some("high".into());
        submit_and_wait(&cm, high);

        let pending = cm.pending_jobs();
        let pending_refs: Vec<&Job> = pending.iter().collect();
        let partitions = cm.get_partitions();
        crate::scheduler_loop::try_preempt(&cm, &partitions, &pending_refs, &cm.config().scheduler)
            .await;

        // low job must still be running — "high" QOS is not allowed to preempt "low"
        assert_eq!(
            cm.get_job(low_id).unwrap().state,
            JobState::Running,
            "preemption must be blocked when pending QOS allow-list does not include running QOS"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qos_preempt_hierarchy_allows_preemption_when_in_allow_list() {
        // With preempt_type = qos_priority and "high" explicitly listing "low"
        // in its preempt allow-list, the preemption must fire.
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].preempt_mode = "cancel".into();
        config.scheduler.preempt_type = spur_core::partition::PreemptType::QosPriority;
        let cm = test_cluster_with_config(&dir, config).await;
        register_node(&cm, "n1", 8, 16000);

        cm.qos_cache().insert(Qos {
            name: "low".into(),
            priority: 100,
            ..Default::default()
        });
        cm.qos_cache().insert(Qos {
            name: "high".into(),
            priority: 10000,
            preempt: vec!["low".into()],
            ..Default::default()
        });

        let mut low = basic_spec("low");
        low.qos = Some("low".into());
        let low_id = submit_and_wait(&cm, low);
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            low_id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, low_id, JobState::Running);

        let mut high = basic_spec("high");
        high.qos = Some("high".into());
        submit_and_wait(&cm, high);

        let pending = cm.pending_jobs();
        let pending_refs: Vec<&Job> = pending.iter().collect();
        let partitions = cm.get_partitions();
        crate::scheduler_loop::try_preempt(&cm, &partitions, &pending_refs, &cm.config().scheduler)
            .await;

        settle(&cm, low_id, JobState::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_exempt_time_protects_recently_started_job() {
        // A job that started less than preempt_exempt_time seconds ago must not
        // be preempted even when the priority gap is large enough.
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.partitions[0].preempt_mode = "cancel".into();
        // Set a very large exempt window so a job started "just now" is always protected.
        config.scheduler.preempt_exempt_time = 3600;
        let cm = test_cluster_with_config(&dir, config).await;
        register_node(&cm, "n1", 8, 16000);

        let mut low = basic_spec("low");
        low.priority = Some(100);
        let low_id = submit_and_wait(&cm, low);
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            low_id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, low_id, JobState::Running);

        // Confirm the job has a start_time (required for exempt check to fire).
        let low_job = cm.get_job(low_id).unwrap();
        assert!(
            low_job.start_time.is_some(),
            "running job must have a start_time"
        );

        let mut high = basic_spec("high");
        high.priority = Some(10_000);
        submit_and_wait(&cm, high);

        let pending = cm.pending_jobs();
        let pending_refs: Vec<&Job> = pending.iter().collect();
        let partitions = cm.get_partitions();
        crate::scheduler_loop::try_preempt(&cm, &partitions, &pending_refs, &cm.config().scheduler)
            .await;

        // low job must still be running — it was started moments ago and is within the exempt window
        assert_eq!(
            cm.get_job(low_id).unwrap().state,
            JobState::Running,
            "job started within preempt_exempt_time window must not be preempted"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_cancel_unblocks_afterok_and_afterany_dependents() {
        // A cancel-preempted job lands in Cancelled (terminal), so afterok
        // and afterany dependents must stop waiting and resolve their outcome.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);

        let parent_id = run_job_on(&cm, "parent", "worker1");
        cm.preempt_job_with_provenance(parent_id, PreemptMode::Cancel, Some(99), None)
            .unwrap();
        settle(&cm, parent_id, JobState::Cancelled);

        // afterok: parent exited non-zero (preempted) → unsatisfiable; watcher cancels it.
        let mut afterok_child = basic_spec("afterok-child");
        afterok_child.dependency = vec![format!("afterok:{parent_id}")];
        let afterok_id = submit_and_wait(&cm, afterok_child);

        // afterany: fires regardless of exit status → satisfied once parent is terminal.
        let mut afterany_child = basic_spec("afterany-child");
        afterany_child.dependency = vec![format!("afterany:{parent_id}")];
        let afterany_id = submit_and_wait(&cm, afterany_child);

        cm.cancel_unsatisfiable_dependency_jobs();

        // afterok cancelled: parent's non-zero exit makes it permanently unsatisfiable.
        assert_eq!(
            cm.get_job(afterok_id).unwrap().state,
            JobState::Cancelled,
            "afterok dependent must be cancelled after parent was preempted"
        );

        // afterany unblocked: parent is terminal, dependency is satisfied.
        // pending_jobs() returns only schedulable (non-blocked) jobs.
        assert!(
            cm.pending_jobs().iter().any(|j| j.job_id == afterany_id),
            "afterany dependent must be schedulable once parent is terminal"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_uses_candidate_qos_preempt_mode_over_partition() {
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        // Partition says Cancel; the candidate's QoS overrides it to Suspend.
        config.partitions[0].preempt_mode = "cancel".into();
        let cm = test_cluster_with_config(&dir, config).await;
        register_node(&cm, "n1", 8, 16000);

        cm.qos_cache().insert(Qos {
            name: "suspend-me".into(),
            preempt_mode: spur_core::accounting::QosPreemptMode::Suspend,
            ..Default::default()
        });

        let mut low = basic_spec("low");
        low.priority = Some(100);
        low.qos = Some("suspend-me".into());
        let low_id = submit_and_wait(&cm, low);
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            low_id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, low_id, JobState::Running);

        let mut high = basic_spec("high");
        high.priority = Some(10_000);
        let high_id = submit_and_wait(&cm, high);
        let high_job = cm.get_job(high_id).unwrap();
        let partitions = cm.get_partitions();

        crate::scheduler_loop::try_preempt(&cm, &partitions, &[&high_job], &cm.config().scheduler)
            .await;

        // Suspended, not Cancelled: proves the QoS override reached the real
        // preemption action, not just the pure job_preempt_mode() decision.
        settle(&cm, low_id, JobState::Suspended);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn purge_expired_holds_pending_reservation_jobs() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        cm.apply_operation(&WalOperation::ReservationCreate {
            reservation: Reservation {
                name: "r1".into(),
                start_time: now - chrono::Duration::hours(2),
                end_time: now - chrono::Duration::minutes(1),
                nodes: vec!["n1".into()],
                accounts: Vec::new(),
                users: vec!["testuser".into()],
                flags: Default::default(),
                owner: String::new(),
            },
        });

        let mut spec = basic_spec("resv-pending");
        spec.reservation = Some("r1".into());
        let job_id = submit_and_wait(&cm, spec);

        cm.purge_expired_reservations();
        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.pending_reason, PendingReason::ReservationDeleted);
        assert_eq!(job.priority, 0);
        assert!(cm.get_reservations().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_reservation_create_update_delete() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        let res = Reservation {
            name: "r1".into(),
            start_time: now,
            end_time: now + chrono::Duration::hours(1),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: vec!["alice".into()],
            flags: Default::default(),
            owner: String::new(),
        };
        cm.apply_operation(&WalOperation::ReservationCreate {
            reservation: res.clone(),
        });
        assert_eq!(cm.get_reservations().len(), 1);
        assert_eq!(cm.get_reservations()[0].name, "r1");

        cm.apply_operation(&WalOperation::ReservationUpdate {
            name: "r1".into(),
            duration_minutes: 120,
            add_nodes: Vec::new(),
            remove_nodes: Vec::new(),
            add_users: vec!["bob".into()],
            remove_users: Vec::new(),
            add_accounts: Vec::new(),
            remove_accounts: Vec::new(),
        });
        let updated = cm
            .get_reservations()
            .into_iter()
            .find(|r| r.name == "r1")
            .unwrap();
        assert!(updated.users.contains(&"bob".into()));
        assert_eq!(
            updated.end_time,
            updated.start_time + chrono::Duration::minutes(120)
        );

        cm.apply_operation(&WalOperation::ReservationDelete { name: "r1".into() });
        assert!(cm.get_reservations().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_reservation_create_idempotent() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        let res = Reservation {
            name: "r1".into(),
            start_time: now,
            end_time: now + chrono::Duration::hours(1),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: vec!["alice".into()],
            flags: Default::default(),
            owner: String::new(),
        };
        cm.apply_operation(&WalOperation::ReservationCreate {
            reservation: res.clone(),
        });
        cm.apply_operation(&WalOperation::ReservationCreate { reservation: res });
        assert_eq!(cm.get_reservations().len(), 1);
        assert_eq!(cm.get_reservations()[0].name, "r1");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_reservation_create_keeps_single_entry() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        let base = Reservation {
            name: "r1".into(),
            start_time: now,
            end_time: now + chrono::Duration::hours(1),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: vec!["alice".into()],
            flags: Default::default(),
            owner: String::new(),
        };
        let cm1 = cm.clone();
        let cm2 = cm.clone();
        let r1 = base.clone();
        let r2 = base;
        let (first, second) = tokio::join!(
            tokio::task::spawn_blocking(move || cm1.create_reservation(r1)),
            tokio::task::spawn_blocking(move || cm2.create_reservation(r2)),
        );
        let outcomes = [first.unwrap(), second.unwrap()];
        assert_eq!(
            outcomes.iter().filter(|r| r.is_ok()).count(),
            1,
            "exactly one concurrent create must succeed"
        );
        assert_eq!(cm.get_reservations().len(), 1);
        assert_eq!(cm.get_reservations()[0].name, "r1");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reservation_survives_wal_replay() {
        let dir = TempDir::new().unwrap();
        {
            let cm = test_cluster(&dir).await;
            register_node(&cm, "n1", 8, 16000);
            let now = chrono::Utc::now();
            cm.create_reservation(Reservation {
                name: "r1".into(),
                start_time: now,
                end_time: now + chrono::Duration::hours(1),
                nodes: vec!["n1".into()],
                accounts: Vec::new(),
                users: vec!["alice".into()],
                flags: Default::default(),
                owner: String::new(),
            })
            .unwrap();
            assert_eq!(cm.get_reservations().len(), 1);
        }

        let cm2 = test_cluster(&dir).await;
        wait_for("reservation replayed from WAL", || {
            cm2.get_reservations().iter().any(|r| r.name == "r1")
        });
        assert_eq!(cm2.get_reservations().len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_hold_jobs_reservation_delete_does_not_permanently_block_job() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        cm.create_reservation(Reservation {
            name: "r1".into(),
            start_time: now - chrono::Duration::minutes(5),
            end_time: now + chrono::Duration::hours(2),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: vec!["testuser".into()],
            flags: spur_core::reservation::ReservationFlags {
                no_hold_jobs: true,
                ..Default::default()
            },
            owner: String::new(),
        })
        .unwrap();

        let mut spec = basic_spec("resv-pending");
        spec.reservation = Some("r1".into());
        let job_id = submit_and_wait(&cm, spec);

        cm.delete_reservation("r1").unwrap();
        wait_for("reservation deleted", || cm.get_reservations().is_empty());

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.spec.reservation, None);
        assert_ne!(job.pending_reason, PendingReason::Held);
        assert_ne!(job.priority, 0);
        assert!(
            cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "job must remain schedulable after no_hold_jobs delete"
        );
    }

    /// Authorization is the RPC boundary's, not the state machine's: an operator may manage any
    /// reservation, so `owner` is attribution only.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reservation_management_does_not_consult_the_owner() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        cm.create_reservation(Reservation {
            name: "r1".into(),
            start_time: now - chrono::Duration::minutes(5),
            end_time: now + chrono::Duration::hours(2),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: Vec::new(),
            flags: Default::default(),
            owner: "alice".into(),
        })
        .unwrap();

        cm.update_reservation("r1", 30, &[], &[], &[], &[], &[], &[])
            .expect("update must not depend on the recorded owner");
        cm.delete_reservation("r1")
            .expect("delete must not depend on the recorded owner");
        assert!(
            cm.get_reservations().is_empty(),
            "deleted reservation must not survive"
        );

        let err = cm.delete_reservation("r1").unwrap_err();
        assert!(
            matches!(err, ReservationError::NotFound(_)),
            "deleting a reservation that is gone must report not-found, got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_reservation_rejects_non_positive_span() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();

        // A zero-length span (duration 0) would be created but is immediately
        // expired, so the scheduler purge loop would drop it before it is ever
        // visible. Reject it up front instead.
        let err = cm
            .create_reservation(Reservation {
                name: "zero".into(),
                start_time: now,
                end_time: now,
                nodes: vec!["n1".into()],
                accounts: Vec::new(),
                users: Vec::new(),
                flags: Default::default(),
                owner: "root".into(),
            })
            .unwrap_err();
        assert!(
            matches!(err, ReservationError::InvalidArgument(_)),
            "zero-length reservation must be rejected, got {err:?}"
        );
        assert!(
            cm.get_reservations().is_empty(),
            "rejected reservation must not persist"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_hold_jobs_purge_does_not_permanently_block_job() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        cm.apply_operation(&WalOperation::ReservationCreate {
            reservation: Reservation {
                name: "r1".into(),
                start_time: now - chrono::Duration::hours(2),
                end_time: now - chrono::Duration::minutes(1),
                nodes: vec!["n1".into()],
                accounts: Vec::new(),
                users: vec!["testuser".into()],
                flags: spur_core::reservation::ReservationFlags {
                    no_hold_jobs: true,
                    ..Default::default()
                },
                owner: String::new(),
            },
        });

        let mut spec = basic_spec("resv-pending");
        spec.reservation = Some("r1".into());
        let job_id = submit_and_wait(&cm, spec);

        cm.purge_expired_reservations();
        wait_for("reservation purged", || cm.get_reservations().is_empty());

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.spec.reservation, None);
        assert!(
            cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "job must remain schedulable after no_hold_jobs purge"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reservation_fence_reason_requires_all_candidates_blocked() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        register_node(&cm, "n2", 8, 16000);
        let now = chrono::Utc::now();
        cm.create_reservation(Reservation {
            name: "r1".into(),
            start_time: now - chrono::Duration::minutes(5),
            end_time: now + chrono::Duration::hours(2),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: vec!["alice".into()],
            flags: Default::default(),
            owner: String::new(),
        })
        .unwrap();

        let job_id = submit_and_wait(&cm, basic_spec("fence"));
        cm.refresh_pending_reasons();
        let job = cm.get_job(job_id).unwrap();
        assert_ne!(
            job.pending_reason,
            PendingReason::ReqNodeNotAvail,
            "n2 is an unblocked candidate; must not fence the job"
        );
        assert_ne!(job.pending_reason, PendingReason::ReservedMaintenance);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_state_change_held_pending_apply_is_atomic() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let job_id = submit_and_wait(&cm, basic_spec("hold-atomic"));
        cm.apply_operation(&WalOperation::job_state_change(
            job_id,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobComplete {
            job_id,
            exit_code: -1,
            state: JobState::Preempted,
        });
        for _ in 0..5 {
            cm.apply_operation(&WalOperation::job_state_change(
                job_id,
                JobState::Preempted,
                JobState::Pending,
            ));
            cm.apply_operation(&WalOperation::job_state_change(
                job_id,
                JobState::Pending,
                JobState::Running,
            ));
            cm.apply_operation(&WalOperation::JobComplete {
                job_id,
                exit_code: -1,
                state: JobState::Preempted,
            });
        }
        assert_eq!(cm.get_job(job_id).unwrap().requeue_count, 5);

        cm.apply_operation(&WalOperation::job_state_change_held_pending(
            job_id,
            JobState::Preempted,
            PendingReason::JobHoldMaxRequeue,
        ));
        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.priority, 0);
        assert_eq!(job.pending_reason, PendingReason::JobHoldMaxRequeue);
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "hold must apply priority and reason in one WAL entry"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn requeue_preempted_job_holds_at_max_requeue() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let job_id = submit_and_wait(&cm, basic_spec("preempt-me"));
        cm.apply_operation(&WalOperation::job_state_change(
            job_id,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobComplete {
            job_id,
            exit_code: -1,
            state: JobState::Preempted,
        });
        for _ in 0..5 {
            cm.apply_operation(&WalOperation::job_state_change(
                job_id,
                JobState::Preempted,
                JobState::Pending,
            ));
            cm.apply_operation(&WalOperation::job_state_change(
                job_id,
                JobState::Pending,
                JobState::Running,
            ));
            cm.apply_operation(&WalOperation::JobComplete {
                job_id,
                exit_code: -1,
                state: JobState::Preempted,
            });
        }
        assert_eq!(cm.get_job(job_id).unwrap().requeue_count, 5);

        cm.hold_job_at_max_requeue(job_id).unwrap();
        wait_for("job held at max requeue", || {
            cm.get_job(job_id).is_some_and(|j| {
                j.state == JobState::Pending && j.pending_reason == PendingReason::JobHoldMaxRequeue
            })
        });
        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.priority, 0);
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "held job must not be schedulable"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reservation_delete_hold_survives_wal_replay() {
        let dir = TempDir::new().unwrap();
        {
            let cm = test_cluster(&dir).await;
            register_node(&cm, "n1", 8, 16000);
            let now = chrono::Utc::now();
            cm.create_reservation(Reservation {
                name: "r1".into(),
                start_time: now - chrono::Duration::minutes(5),
                end_time: now + chrono::Duration::hours(2),
                nodes: vec!["n1".into()],
                accounts: Vec::new(),
                users: vec!["testuser".into()],
                flags: Default::default(),
                owner: String::new(),
            })
            .unwrap();

            let mut spec = basic_spec("resv-hold");
            spec.reservation = Some("r1".into());
            let job_id = submit_and_wait(&cm, spec);
            cm.delete_reservation("r1").unwrap();
            wait_for("job held after delete", || {
                cm.get_job(job_id).is_some_and(|j| {
                    j.pending_reason == PendingReason::ReservationDeleted && j.priority == 0
                })
            });
        }

        let cm2 = test_cluster(&dir).await;
        wait_for("held job replayed from WAL", || {
            cm2.jobs
                .read()
                .values()
                .any(|j| j.pending_reason == PendingReason::ReservationDeleted && j.priority == 0)
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reservation_delete_no_hold_survives_wal_replay() {
        let dir = TempDir::new().unwrap();
        let job_id;
        {
            let cm = test_cluster(&dir).await;
            register_node(&cm, "n1", 8, 16000);
            let now = chrono::Utc::now();
            cm.create_reservation(Reservation {
                name: "r1".into(),
                start_time: now - chrono::Duration::minutes(5),
                end_time: now + chrono::Duration::hours(2),
                nodes: vec!["n1".into()],
                accounts: Vec::new(),
                users: vec!["testuser".into()],
                flags: spur_core::reservation::ReservationFlags {
                    no_hold_jobs: true,
                    ..Default::default()
                },
                owner: String::new(),
            })
            .unwrap();

            let mut spec = basic_spec("resv-no-hold");
            spec.reservation = Some("r1".into());
            job_id = submit_and_wait(&cm, spec);
            cm.delete_reservation("r1").unwrap();
            wait_for("reservation detached from job", || {
                cm.get_job(job_id)
                    .is_some_and(|j| j.spec.reservation.is_none() && j.priority > 0)
            });
        }

        let cm2 = test_cluster(&dir).await;
        wait_for("detached job replayed from WAL", || {
            cm2.get_job(job_id)
                .is_some_and(|j| j.spec.reservation.is_none() && j.priority > 0)
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn release_job_after_reservation_delete_unblocks_job() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let now = chrono::Utc::now();
        cm.create_reservation(Reservation {
            name: "r1".into(),
            start_time: now - chrono::Duration::minutes(5),
            end_time: now + chrono::Duration::hours(2),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: vec!["testuser".into()],
            flags: Default::default(),
            owner: String::new(),
        })
        .unwrap();

        let mut spec = basic_spec("release-me");
        spec.reservation = Some("r1".into());
        let job_id = submit_and_wait(&cm, spec);
        cm.delete_reservation("r1").unwrap();
        wait_for("job held after delete", || {
            cm.get_job(job_id).is_some_and(|j| {
                j.pending_reason == PendingReason::ReservationDeleted && j.priority == 0
            })
        });
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "held job must not be schedulable before release"
        );

        cm.release_job(job_id).unwrap();
        wait_for("job released after reservation delete", || {
            cm.get_job(job_id).is_some_and(|j| {
                j.spec.reservation.is_none()
                    && j.priority > 0
                    && j.pending_reason == PendingReason::Priority
            })
        });
        assert!(
            cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "released job must be schedulable"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_preserves_reservation_deleted_reason() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let mut spec = basic_spec("resv-deleted");
        spec.reservation = Some("gone".into());
        let job_id = submit_and_wait(&cm, spec);
        {
            let mut jobs = cm.jobs.write();
            jobs.get_mut(&job_id).unwrap().pending_reason = PendingReason::ReservationDeleted;
            jobs.get_mut(&job_id).unwrap().priority = 0;
        }

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::ReservationDeleted
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preempt_requeue_never_holds_job_regardless_of_repeat_count() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);
        let max = cm.config().controller.max_batch_requeue;

        let job_id = run_job_on(&cm, "chronic-preempt", "worker1");
        for _ in 0..(max + 3) {
            cm.preempt_job_with_provenance(job_id, PreemptMode::Requeue, Some(99), None)
                .unwrap();
            settle(&cm, job_id, JobState::Pending);
            {
                let mut jobs = cm.jobs.write();
                jobs.get_mut(&job_id).unwrap().spec.begin_time =
                    Some(Utc::now() - chrono::Duration::seconds(1));
            }
            let resources = scalar_alloc(2, 4000);
            cm.start_job(
                job_id,
                vec!["worker1".into()],
                resources.clone(),
                per_node_for(&["worker1"], resources),
            )
            .unwrap();
            settle(&cm, job_id, JobState::Running);
        }
        cm.preempt_job_with_provenance(job_id, PreemptMode::Requeue, Some(99), None)
            .unwrap();
        settle(&cm, job_id, JobState::Pending);

        let job = cm.get_job(job_id).unwrap();
        assert!(
            job.preempt_requeue_count > max,
            "job must have been preempted more times than max_batch_requeue"
        );
        assert_eq!(
            job.requeue_count, 0,
            "repeated preemption must not touch the failure-requeue counter"
        );
        assert_ne!(
            job.pending_reason,
            PendingReason::JobHoldMaxRequeue,
            "a chronically-preempted but otherwise healthy job must never be held"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chronic_preemption_does_not_exhaust_failure_requeue_budget() {
        // A job preempted max_batch_requeue times must still get a full
        // failure-requeue budget: the two counters are independent.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "worker1", 8, 16000);
        let max = cm.config().controller.max_batch_requeue;

        let mut spec = basic_spec("chronic-then-timeout");
        spec.requeue = true;
        let job_id = submit_and_wait(&cm, spec);
        let resources = scalar_alloc(2, 4000);

        for _ in 0..max {
            cm.start_job(
                job_id,
                vec!["worker1".into()],
                resources.clone(),
                per_node_for(&["worker1"], resources.clone()),
            )
            .unwrap();
            settle(&cm, job_id, JobState::Running);
            cm.preempt_job_with_provenance(job_id, PreemptMode::Requeue, Some(99), None)
                .unwrap();
            settle(&cm, job_id, JobState::Pending);
            {
                let mut jobs = cm.jobs.write();
                jobs.get_mut(&job_id).unwrap().spec.begin_time =
                    Some(Utc::now() - chrono::Duration::seconds(1));
            }
        }
        assert_eq!(cm.get_job(job_id).unwrap().preempt_requeue_count, max);
        assert_eq!(cm.get_job(job_id).unwrap().requeue_count, 0);

        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);
        cm.complete_job(job_id, -1, JobState::Timeout).unwrap();

        settle(&cm, job_id, JobState::Pending);
        let job = cm.get_job(job_id).unwrap();
        assert_eq!(
            job.requeue_count, 1,
            "the first genuine failure must count, unaffected by prior preemptions"
        );
        assert_ne!(
            job.pending_reason,
            PendingReason::JobHoldMaxRequeue,
            "one timeout after chronic preemption must not exhaust the failure budget"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_partial_dispatch_failure_backs_off_like_a_total_one() {
        // Both launch-failure paths must throttle. Without this, a 2-node job
        // with one broken node evicts, requeues immediately, lands on the same
        // node and burns its whole requeue budget in seconds.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        register_node(&cm, "n2", 4, 8000);

        let mut spec = basic_spec("partial-backoff");
        spec.num_nodes = 2;
        spec.requeue = true;
        let job_id = submit_and_wait(&cm, spec);

        let alloc = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["n1".into(), "n2".into()],
            scalar_alloc(4, 8000),
            per_node_for(&["n1", "n2"], alloc),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        cm.evict_job(job_id).unwrap();
        settle(&cm, job_id, JobState::Pending);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.pending_reason, PendingReason::JobLaunchFailure);
        assert!(
            job.spec.begin_time.is_some_and(|begin| begin > Utc::now()),
            "a launch-failure requeue must carry a future hold"
        );
        assert!(
            !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
            "the hold must keep the job out of the very next dispatch"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_node_failure_requeue_still_retries_immediately() {
        // The backoff is scoped to launch failures. A job evicted because its
        // node died has nothing to back off from: the node is already out of the
        // candidate set, so delaying the retry only wastes capacity.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        register_node(&cm, "n2", 4, 8000);

        let mut spec = basic_spec("nodedown-requeue");
        spec.requeue = true;
        let job_id = submit_and_wait(&cm, spec);

        let alloc = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["n1".into()],
            alloc.clone(),
            per_node_for(&["n1"], alloc),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        if let Some(node) = cm.nodes.write().get_mut("n1") {
            node.last_heartbeat = Some(Utc::now() - chrono::Duration::seconds(200));
        }
        cm.check_node_health(90, super::MarkDownPolicy::Allowed);
        settle(&cm, job_id, JobState::Pending);

        let job = cm.get_job(job_id).unwrap();
        assert_eq!(job.pending_reason, PendingReason::None);
        assert!(job.spec.begin_time.is_none(), "no hold on a node failure");
        assert!(cm.pending_jobs().iter().any(|j| j.job_id == job_id));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn maybe_requeue_holds_at_max_for_timeout() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        let mut spec = basic_spec("timeout-requeue");
        spec.requeue = true;
        let job_id = submit_and_wait(&cm, spec);
        let alloc = scalar_alloc(1, 1000);
        let nodes = vec!["n1".into()];
        let per_node = per_node_for(&["n1"], alloc.clone());

        for _ in 0..5 {
            cm.start_job(job_id, nodes.clone(), alloc.clone(), per_node.clone())
                .unwrap();
            settle(&cm, job_id, JobState::Running);
            cm.complete_job(job_id, -1, JobState::Timeout).unwrap();
            settle(&cm, job_id, JobState::Pending);
        }
        assert_eq!(cm.get_job(job_id).unwrap().requeue_count, 5);

        cm.start_job(job_id, nodes, alloc, per_node).unwrap();
        settle(&cm, job_id, JobState::Running);
        cm.complete_job(job_id, -1, JobState::Timeout).unwrap();
        wait_for("job held at max requeue after timeout", || {
            cm.get_job(job_id).is_some_and(|j| {
                j.state == JobState::Pending && j.pending_reason == PendingReason::JobHoldMaxRequeue
            })
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn release_job_resets_requeue_count_after_max_hold() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let job_id = submit_and_wait(&cm, basic_spec("release-me"));
        cm.apply_operation(&WalOperation::job_state_change(
            job_id,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobComplete {
            job_id,
            exit_code: -1,
            state: JobState::Preempted,
        });
        for _ in 0..5 {
            cm.apply_operation(&WalOperation::job_state_change(
                job_id,
                JobState::Preempted,
                JobState::Pending,
            ));
            cm.apply_operation(&WalOperation::job_state_change(
                job_id,
                JobState::Pending,
                JobState::Running,
            ));
            cm.apply_operation(&WalOperation::JobComplete {
                job_id,
                exit_code: -1,
                state: JobState::Preempted,
            });
        }
        cm.hold_job_at_max_requeue(job_id).unwrap();
        wait_for("job held at max requeue", || {
            cm.get_job(job_id)
                .is_some_and(|j| j.pending_reason == PendingReason::JobHoldMaxRequeue)
        });
        assert_eq!(cm.get_job(job_id).unwrap().requeue_count, 5);

        cm.release_job(job_id).unwrap();
        wait_for("release resets requeue budget", || {
            cm.get_job(job_id).is_some_and(|j| {
                j.requeue_count == 0
                    && j.priority > 0
                    && j.pending_reason == PendingReason::Priority
            })
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_preempted_job_uses_cancelled_state() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let job_id = submit_and_wait(&cm, basic_spec("cancel-preempted"));

        cm.apply_operation(&WalOperation::job_state_change(
            job_id,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobComplete {
            job_id,
            exit_code: -1,
            state: JobState::Preempted,
        });
        assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Preempted);

        cm.cancel_job(job_id, "testuser").unwrap();
        wait_for("preempted job cancelled", || {
            cm.get_job(job_id)
                .is_some_and(|j| j.state == JobState::Cancelled)
        });
        let job = cm.get_job(job_id).unwrap();
        assert_ne!(job.pending_reason, PendingReason::JobHoldMaxRequeue);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_licenses_reason() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let mut spec = basic_spec("lic");
        // Request a license with an empty cluster pool -> shortfall.
        spec.gres = vec!["license:flexlm:1".into()];
        let job_id = submit_and_wait(&cm, spec);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::Licenses
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_qos_reason_from_cache() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        // Seed the cache with a QoS that caps wall time at 1 min, then submit a
        // job to that QoS asking for more — the specific QOS reason must surface
        // through resolve_qos -> qos_block_with (not the old limitless default).
        cm.qos_cache().insert(Qos {
            name: "short".into(),
            limits: spur_core::accounting::QosLimits {
                max_wall_minutes: Some(1),
                ..Default::default()
            },
            ..Default::default()
        });
        let mut spec = basic_spec("qos");
        spec.qos = Some("short".into());
        spec.time_limit = Some(chrono::Duration::hours(1));
        let job_id = submit_and_wait(&cm, spec);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::QosMaxWallDurationPerJobLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_qos_grp_cpu_reason() {
        // GrpCPU aggregates across all running jobs in the QOS: a running 4-cpu
        // job fills a grp_tres cpu=4 cap, so the next job in the same QOS blocks
        // with QOSGrpCpuLimit (the group reason, not the per-job/per-user one).
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let mut grp = TresRecord::new();
        grp.set(TresType::Cpu, 4);
        cm.qos_cache().insert(Qos {
            name: "grp".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        let mut s1 = basic_spec("g1");
        s1.qos = Some("grp".into());
        s1.num_tasks = 4;
        let j1 = submit_and_wait(&cm, s1);
        let res = scalar_alloc(4, 1000);
        cm.start_job(
            j1,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, j1, JobState::Running);

        let mut s2 = basic_spec("g2");
        s2.qos = Some("grp".into());
        s2.num_tasks = 1;
        let j2 = submit_and_wait(&cm, s2);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(j2).unwrap().pending_reason,
            PendingReason::QosGrpCpuLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn association_default_qos_reaches_real_enforcement() {
        // A default isn't just cosmetic: it's subject to the QOS's limits
        // exactly as if --qos had been passed explicitly.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        cm.qos_cache().insert(Qos {
            name: "highprio".into(),
            limits: spur_core::accounting::QosLimits {
                max_jobs_per_user: Some(1),
                ..Default::default()
            },
            ..Default::default()
        });
        cm.association_cache()
            .insert_default_qos("testuser", "research", "highprio");

        let mut s1 = basic_spec("h1");
        s1.account = Some("research".into());
        let j1 = submit_and_wait(&cm, s1);
        // The default was resolved and baked into the job at submission —
        // not merely applied ephemerally during scheduling.
        assert_eq!(
            cm.get_job(j1).unwrap().spec.qos.as_deref(),
            Some("highprio")
        );

        let res = scalar_alloc(1, 1000);
        cm.start_job(
            j1,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, j1, JobState::Running);

        let mut s2 = basic_spec("h2");
        s2.account = Some("research".into());
        let j2 = submit_and_wait(&cm, s2);
        assert_eq!(
            cm.get_job(j2).unwrap().spec.qos.as_deref(),
            Some("highprio")
        );

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(j2).unwrap().pending_reason,
            PendingReason::QoSMaxJobsPerUser
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_default_qos_reaches_real_enforcement() {
        // A cluster fallback QOS must bind a no-qos job and its limits must
        // actually block a second job — end to end.
        let dir = TempDir::new().unwrap();
        let mut config = test_config();
        config.accounting.default_qos = "normal".into();
        let cm = test_cluster_with_config(&dir, config).await;
        register_node(&cm, "n1", 8, 16000);
        cm.qos_cache().insert(Qos {
            name: "normal".into(),
            limits: spur_core::accounting::QosLimits {
                max_jobs_per_user: Some(1),
                ..Default::default()
            },
            ..Default::default()
        });

        let j1 = submit_and_wait(&cm, basic_spec("d1"));
        assert_eq!(
            cm.get_job(j1).unwrap().spec.qos.as_deref(),
            Some("normal"),
            "no-qos job must be bound to the cluster fallback at submit"
        );
        let res = scalar_alloc(1, 1000);
        cm.start_job(
            j1,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, j1, JobState::Running);

        let j2 = submit_and_wait(&cm, basic_spec("d2"));
        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(j2).unwrap().pending_reason,
            PendingReason::QoSMaxJobsPerUser,
            "the fallback QOS's limits must actually enforce"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_account_max_jobs_reason() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        cm.association_cache().insert_limits(
            "testuser",
            "research",
            spur_core::accounting::AccountLimits {
                max_running_jobs: Some(1),
                ..Default::default()
            },
        );

        let mut s1 = basic_spec("a1");
        s1.account = Some("research".into());
        let j1 = submit_and_wait(&cm, s1);
        let res = scalar_alloc(1, 1000);
        cm.start_job(
            j1,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, j1, JobState::Running);

        let mut s2 = basic_spec("a2");
        s2.account = Some("research".into());
        let j2 = submit_and_wait(&cm, s2);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(j2).unwrap().pending_reason,
            PendingReason::AssocMaxJobsLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_account_max_submit_jobs_reason() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        // Submit within limits, then lower the cap: this exercises the
        // scheduler safety-net path (the submit gate would reject up front, so
        // it can only be reached when the limit is tightened after submission).
        cm.association_cache()
            .insert_association("testuser", "research");

        let mut s1 = basic_spec("b1");
        s1.account = Some("research".into());
        let j1 = submit_and_wait(&cm, s1);

        let mut s2 = basic_spec("b2");
        s2.account = Some("research".into());
        let j2 = submit_and_wait(&cm, s2);

        cm.association_cache().insert_limits(
            "testuser",
            "research",
            spur_core::accounting::AccountLimits {
                max_submit_jobs: Some(1),
                ..Default::default()
            },
        );

        cm.refresh_pending_reasons();
        // j1 alone is within the limit (max_submit_jobs=1) and must not be
        // blocked by counting itself; only j2, which pushes the count over
        // the cap, should be blocked.
        assert_eq!(cm.get_job(j1).unwrap().pending_reason, PendingReason::None);
        assert_eq!(
            cm.get_job(j2).unwrap().pending_reason,
            PendingReason::AssocMaxSubmitJobLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_qos_max_submit_jobs_reason() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        // Submit under an uncapped QOS, then tighten it: the submit gate would
        // reject up front, so the scheduler tagging path is only reachable when
        // the limit is lowered after submission.
        cm.qos_cache().insert(Qos {
            name: "capped".into(),
            ..Default::default()
        });

        let mut s1 = basic_spec("c1");
        s1.qos = Some("capped".into());
        let j1 = submit_and_wait(&cm, s1);

        let mut s2 = basic_spec("c2");
        s2.qos = Some("capped".into());
        let j2 = submit_and_wait(&cm, s2);

        cm.qos_cache().insert(Qos {
            name: "capped".into(),
            limits: spur_core::accounting::QosLimits {
                max_submit_jobs_per_user: Some(1),
                ..Default::default()
            },
            ..Default::default()
        });

        cm.refresh_pending_reasons();
        // j1 alone is within the limit (max_submit_jobs_per_user=1) and must
        // not be blocked by counting itself; only j2, which pushes the count
        // over the cap, should be blocked.
        assert_eq!(cm.get_job(j1).unwrap().pending_reason, PendingReason::None);
        assert_eq!(
            cm.get_job(j2).unwrap().pending_reason,
            PendingReason::QosMaxSubmitJobPerUserLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_sets_account_grp_cpu_reason_across_users() {
        // GrpTRES aggregates across every user in the account, not just the
        // requester: a different user's running job fills the cap so the next
        // job in the same account blocks with AssocGrpCpuLimit.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let mut grp = TresRecord::new();
        grp.set(TresType::Cpu, 4);
        cm.association_cache().insert_limits(
            "alice",
            "research",
            spur_core::accounting::AccountLimits {
                grp_tres: Some(grp.clone()),
                ..Default::default()
            },
        );
        cm.association_cache().insert_limits(
            "bob",
            "research",
            spur_core::accounting::AccountLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
        );

        let mut s1 = basic_spec("c1");
        s1.user = "alice".into();
        s1.account = Some("research".into());
        s1.num_tasks = 4;
        let j1 = submit_and_wait(&cm, s1);
        let res = scalar_alloc(4, 1000);
        cm.start_job(
            j1,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, j1, JobState::Running);

        let mut s2 = basic_spec("c2");
        s2.user = "bob".into();
        s2.account = Some("research".into());
        s2.num_tasks = 1;
        let j2 = submit_and_wait(&cm, s2);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(j2).unwrap().pending_reason,
            PendingReason::AssocGrpCpuLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn in_pass_qos_grp_node_block_tags_reason_not_none() {
        // Two 1-node jobs, QOS grp_tres node=1, nothing running: the second is
        // blocked only by the first's in-pass reservation. Expect QosGrpNodeLimit.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 1);
        cm.qos_cache().insert(Qos {
            name: "capped".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        let mut s1 = basic_spec("q1");
        s1.qos = Some("capped".into());
        let j1 = submit_and_wait(&cm, s1);
        let mut s2 = basic_spec("q2");
        s2.qos = Some("capped".into());
        let j2 = submit_and_wait(&cm, s2);

        cm.refresh_pending_reasons();
        assert_eq!(cm.get_job(j1).unwrap().pending_reason, PendingReason::None);
        assert_eq!(
            cm.get_job(j2).unwrap().pending_reason,
            PendingReason::QosGrpNodeLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qos_grp_node_admits_job_packable_onto_already_used_nodes() {
        // QOS grp node=4 is fully occupied by two running jobs on disjoint node
        // pairs, each node far below its capacity. A new job needing only 1 more
        // node is admitted by packing onto an already-used node's spare capacity
        // instead of requiring a fifth distinct node.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        for n in ["n1", "n2", "n3", "n4"] {
            register_node(&cm, n, 8, 128000);
        }

        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 4);
        cm.qos_cache().insert(Qos {
            name: "cvs".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        {
            let mut jobs = cm.jobs.write();
            for (id, nodes) in [(101u32, ["n1", "n2"]), (102, ["n3", "n4"])] {
                let mut j = make_running_job(id, &nodes, 1);
                j.spec.num_nodes = 2;
                j.spec.qos = Some("cvs".into());
                jobs.insert(id, j);
            }
        }

        let mut newjob = basic_spec("pack-me");
        newjob.qos = Some("cvs".into());
        newjob.num_nodes = 1;
        let new_id = submit_and_wait(&cm, newjob);

        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        assert!(
            pending.contains(&new_id),
            "job requesting 1 node, packable onto an already-used node with idle \
             capacity, must not be blocked on QOSGrpNodeLimit"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qos_grp_node_still_blocks_when_occupied_nodes_have_no_spare_capacity() {
        // Same grp node=4 shape as the packable case above, but each occupied
        // node is registered with zero capacity — this test harness inserts
        // running jobs directly into the job map without updating node-side
        // allocation, so a genuinely "no headroom" node must be modeled via
        // zero total capacity rather than an exact-fit allocation. The new job
        // must still block on QOSGrpNodeLimit, proving the packing credit only
        // applies when real spare capacity exists.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        for n in ["n1", "n2", "n3", "n4"] {
            register_node(&cm, n, 0, 0);
        }

        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 4);
        cm.qos_cache().insert(Qos {
            name: "cvs".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        {
            let mut jobs = cm.jobs.write();
            for (id, nodes) in [(101u32, ["n1", "n2"]), (102, ["n3", "n4"])] {
                let mut j = make_running_job(id, &nodes, 1);
                j.spec.num_nodes = 2;
                j.spec.qos = Some("cvs".into());
                jobs.insert(id, j);
            }
        }

        let mut newjob = basic_spec("no-room");
        newjob.qos = Some("cvs".into());
        newjob.num_nodes = 1;
        let new_id = submit_and_wait(&cm, newjob);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(new_id).unwrap().pending_reason,
            PendingReason::QosGrpNodeLimit,
            "no occupied node has spare capacity, so the job must still need a \
             5th distinct node and block on the grp cap of 4"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qos_grp_node_partial_reuse_counts_only_the_shortfall() {
        // n1 has spare capacity, n2 has none (zero total capacity — this
        // harness doesn't track node-side allocation for directly-inserted
        // running jobs, so "no room" must be modeled structurally). A job
        // needing 2 nodes can reuse only n1, so it still needs exactly 1 new
        // distinct node — not 0 (crediting both occupied nodes) and not 2
        // (crediting neither).
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 128000);
        register_node(&cm, "n2", 0, 0);

        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 2);
        cm.qos_cache().insert(Qos {
            name: "cvs".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        {
            let mut jobs = cm.jobs.write();
            let mut j = make_running_job(201, &["n1", "n2"], 1);
            j.spec.num_nodes = 2;
            j.spec.qos = Some("cvs".into());
            jobs.insert(201, j);
        }

        let mut newjob = basic_spec("partial-pack");
        newjob.qos = Some("cvs".into());
        newjob.num_nodes = 2;
        newjob.num_tasks = 2; // avoid effective_num_nodes() clamping to num_tasks
        let new_id = submit_and_wait(&cm, newjob);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(new_id).unwrap().pending_reason,
            PendingReason::QosGrpNodeLimit,
            "only n1 has spare capacity to reuse, so the job still needs 1 new \
             distinct node beyond the 2 already occupying the grp cap of 2"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qos_grp_node_packing_credit_actually_places_on_occupied_node() {
        // A 3rd, fully idle node sits alongside the 2 occupied ones the grp
        // node=2 cap counts. Without preferred_nodes, the scheduler's default
        // least-loaded tiebreaker would place the admitted job on the idle
        // node instead of packing onto n1/n2, silently exceeding the true
        // grp cap once running. Assert both the admission credit and the
        // real scheduler's placement land on n1/n2, never n3.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        for n in ["n1", "n2", "n3"] {
            register_node(&cm, n, 8, 128000);
        }

        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 2);
        cm.qos_cache().insert(Qos {
            name: "cvs".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        {
            let mut jobs = cm.jobs.write();
            for (id, nodes) in [(101u32, ["n1"]), (102, ["n2"])] {
                let mut j = make_running_job(id, &nodes, 1);
                j.spec.num_nodes = 1;
                j.spec.qos = Some("cvs".into());
                jobs.insert(id, j);
            }
        }

        let mut newjob = basic_spec("pack-me");
        newjob.qos = Some("cvs".into());
        newjob.num_nodes = 1;
        let new_id = submit_and_wait(&cm, newjob);

        let pending = cm.pending_jobs();
        let job = pending
            .iter()
            .find(|j| j.job_id == new_id)
            .expect("job must be admitted past the grp node cap via packing credit");
        assert_eq!(
            job.preferred_nodes,
            HashSet::from(["n1".to_string(), "n2".to_string()]),
            "admission must credit exactly the occupied nodes with spare capacity"
        );

        let nodes = cm.get_nodes();
        let partitions = cm.get_partitions();
        let reservations = cm.get_reservations();
        let cluster_state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &partitions,
            reservations: &reservations,
            topology: None,
        };
        use spur_sched::traits::Scheduler as _;
        let mut scheduler = spur_sched::backfill::BackfillScheduler::new(10);
        let assignments = scheduler.schedule(&pending, &cluster_state);
        let assignment = assignments
            .iter()
            .find(|a| a.job_id == new_id)
            .expect("scheduler must actually place the admitted job");
        assert!(
            assignment.nodes.iter().all(|n| n == "n1" || n == "n2"),
            "placement must land on an already-occupied node (n1/n2), not the idle n3: {:?}",
            assignment.nodes
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qos_grp_node_same_pass_candidates_do_not_double_credit_one_occupied_node() {
        // Two running jobs already occupy n1 and n2 under a QOS capped at
        // grp_tres node=2 -- the cap is already fully claimed by real running
        // jobs. Only n1 has spare capacity (n2 is modeled at zero total
        // capacity, per this harness's convention for "no room"). Two pending
        // 1-node jobs of the same QOS are submitted together and evaluated in
        // the *same* classify_pending_jobs() pass: only one of them can
        // actually pack onto n1's spare capacity, so the other must be
        // charged for a genuinely new (3rd) distinct node and blocked. If
        // both independently compute a reuse credit for n1 (each unaware the
        // other already claimed it this pass), both are admitted with
        // grp_node_charge=0 and the grp cap of 2 is silently exceeded once
        // real placement resolves the race for n1.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 128000);
        register_node(&cm, "n2", 0, 0);

        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 2);
        cm.qos_cache().insert(Qos {
            name: "cvs".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        {
            let mut jobs = cm.jobs.write();
            for (id, nodes) in [(301u32, ["n1"]), (302, ["n2"])] {
                let mut j = make_running_job(id, &nodes, 1);
                j.spec.num_nodes = 1;
                j.spec.qos = Some("cvs".into());
                jobs.insert(id, j);
            }
        }

        let mut job_a = basic_spec("pack-a");
        job_a.qos = Some("cvs".into());
        job_a.num_nodes = 1;
        let id_a = submit_and_wait(&cm, job_a);

        let mut job_b = basic_spec("pack-b");
        job_b.qos = Some("cvs".into());
        job_b.num_nodes = 1;
        let id_b = submit_and_wait(&cm, job_b);

        let pending = cm.pending_jobs();
        let a = pending.iter().find(|j| j.job_id == id_a);
        let b = pending.iter().find(|j| j.job_id == id_b);

        assert!(
            a.is_some(),
            "the higher-priority candidate must be admitted and credited for reusing n1"
        );
        assert_eq!(
            a.unwrap().preferred_nodes,
            HashSet::from(["n1".to_string()]),
            "the admitted job must be credited with n1 specifically"
        );
        assert!(
            b.is_none(),
            "a second same-pass candidate must not ALSO be credited for n1's \
             already-claimed spare capacity: at most one job can actually \
             pack onto it, so the second must be charged for a genuinely new \
             (3rd) node and blocked at the grp cap of 2"
        );

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(id_b).unwrap().pending_reason,
            PendingReason::QosGrpNodeLimit,
            "the second job must block on the grp node cap rather than being \
             silently admitted on a duplicate reuse credit for n1"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn account_grp_node_same_pass_candidates_do_not_double_credit_one_occupied_node() {
        // Same bug, one layer up: an account association capped at grp_tres
        // node=2, already fully claimed by two real running jobs (only one
        // of which has spare capacity), must not let two same-pass pending
        // jobs both claim reuse credit for that single occupied node.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 128000);
        register_node(&cm, "n2", 0, 0);

        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 2);
        cm.association_cache().insert_limits(
            "testuser",
            "research",
            spur_core::accounting::AccountLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
        );

        {
            let mut jobs = cm.jobs.write();
            for (id, nodes) in [(303u32, ["n1"]), (304, ["n2"])] {
                let mut j = make_running_job(id, &nodes, 1);
                j.spec.num_nodes = 1;
                j.spec.account = Some("research".into());
                jobs.insert(id, j);
            }
        }

        let mut job_a = basic_spec("acct-pack-a");
        job_a.account = Some("research".into());
        job_a.num_nodes = 1;
        let id_a = submit_and_wait(&cm, job_a);

        let mut job_b = basic_spec("acct-pack-b");
        job_b.account = Some("research".into());
        job_b.num_nodes = 1;
        let id_b = submit_and_wait(&cm, job_b);

        let pending = cm.pending_jobs();
        let a = pending.iter().find(|j| j.job_id == id_a);
        let b = pending.iter().find(|j| j.job_id == id_b);

        assert!(
            a.is_some(),
            "the higher-priority candidate must be admitted and credited for reusing n1"
        );
        assert_eq!(
            a.unwrap().preferred_nodes,
            HashSet::from(["n1".to_string()]),
        );
        assert!(
            b.is_none(),
            "a second same-pass candidate must not also be credited for n1's \
             already-claimed spare capacity"
        );

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(id_b).unwrap().pending_reason,
            PendingReason::AssocGrpNodeLimit,
            "the second job must block on the grp node cap rather than being \
             silently admitted on a duplicate reuse credit for n1"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn account_credit_does_not_poison_qos_reuse_computation_for_dual_membership_job() {
        // A job with BOTH an account and a QOS runs account_block_with first,
        // then qos_block_with, on the SAME `&mut Job`. account_block_with
        // credits n1 (a node occupied only by an account-only running job)
        // with a *full* charge=0 for this 1-node job, which immediately makes
        // `job.preferred_nodes = {n1}` fully cover `num_nodes` (1) -- i.e.
        // RESTRICTIVE per `NodePlacement::nodelist_is_additive()`.
        //
        // qos_block_with runs next and internally calls
        // `new_distinct_nodes_needed`, which builds a FRESH
        // `NodePlacement::new(job)` from the job's CURRENT state -- which by
        // now already has `preferred_nodes = {n1}` from the account check.
        // Since that's restrictive, `allows_name` rejects any node not in
        // {n1}, so qos's own occupied node n2 (occupied only by a qos-only
        // running job, with spare capacity, utterly unrelated to the
        // account) gets incorrectly filtered OUT of qos's reusable-node
        // candidates -- even though n2 has nothing to do with the account's
        // credit and QOS's own occupied-node view never included n1.
        //
        // The account's cap is generous (node=5, real=1: crediting an
        // unrelated new node is cheap for it). The QOS cap is tight
        // (node=1, real=1: it has zero headroom for anything except
        // reusing its own already-occupied node n2 with charge=0). The
        // mathematically correct outcome is ADMIT: place the job on n2,
        // giving QOS charge=0 (stays at cap 1) and the account a fresh
        // node (real becomes 2, well under its cap of 5). Instead, the
        // account's irrelevant credit poisons the QOS's own computation,
        // charging it 1 new node it doesn't need, and the job is wrongly
        // blocked on QosGrpNodeLimit -- the exact under-utilization bug
        // class issue #709 (and this whole PR) was meant to fix, now
        // reintroduced via the interaction between the two gates.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 128000);
        register_node(&cm, "n2", 8, 128000);

        let mut acct_grp = TresRecord::new();
        acct_grp.set(TresType::Node, 5);
        cm.association_cache().insert_limits(
            "testuser",
            "acct1",
            spur_core::accounting::AccountLimits {
                grp_tres: Some(acct_grp),
                ..Default::default()
            },
        );
        let mut qos_grp = TresRecord::new();
        qos_grp.set(TresType::Node, 1);
        cm.qos_cache().insert(Qos {
            name: "qos1".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(qos_grp),
                ..Default::default()
            },
            ..Default::default()
        });

        {
            let mut jobs = cm.jobs.write();
            let mut r1 = make_running_job(401, &["n1"], 1);
            r1.spec.num_nodes = 1;
            r1.spec.account = Some("acct1".into());
            jobs.insert(401, r1);
            let mut r2 = make_running_job(402, &["n2"], 1);
            r2.spec.num_nodes = 1;
            r2.spec.qos = Some("qos1".into());
            jobs.insert(402, r2);
        }

        let mut newjob = basic_spec("dual-cap");
        newjob.account = Some("acct1".into());
        newjob.qos = Some("qos1".into());
        newjob.num_nodes = 1;
        let new_id = submit_and_wait(&cm, newjob);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(new_id).unwrap().pending_reason,
            PendingReason::None,
            "the job can be admitted by placing on n2 (QOS reuses it with 0 new \
             nodes, staying at its cap of 1; the account merely gains a new node \
             well within its generous cap of 5) -- it must not be blocked just \
             because the unrelated account credit for n1 happened to run first \
             and made job.preferred_nodes restrictive to {{n1}} before the QOS \
             check ran"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn new_distinct_nodes_needed_does_not_credit_an_unschedulable_node() {
        // n1 is Down (e.g. hardware fault) but still hosts a running QOS job
        // with spare capacity on paper -- this is realistic: taking a node
        // Down/Drain does not kill jobs already running on it. n2 and n3 are
        // healthy idle nodes with plenty of room. QOS grp_tres node=3, real
        // usage is 1 (n1, via the running job) -- enough headroom for 2
        // genuinely new nodes, but NOT enough to also pretend n1's spare
        // capacity is reusable AND need 2 more (that would be 4 total).
        //
        // Before the fix, `new_distinct_nodes_needed` credited n1 as reusable
        // without checking `Node::is_schedulable()` (unlike the real
        // scheduler's `NodePlacement::matches`, which backfill actually uses
        // to filter real candidates): the job would be charged only 1 new
        // node (assuming reuse of n1) and admitted with
        // `job.preferred_nodes = {"n1"}` -- additive, since num_nodes(2) >
        // preferred_nodes.len()(1). Since n1 can never actually be placed on,
        // `BackfillScheduler::find_suitable_nodes`'s additive-nodelist guard
        // (backfill.rs) treats a resource-capable-but-currently-unsuitable
        // listed node as "wait for it, don't skip to unlisted nodes" and
        // returned zero candidates entirely -- starving the job indefinitely
        // even though n2 and n3 sit idle and could satisfy it right now.
        //
        // Fixed: n1 must not be credited at all (charge=2, both of the job's
        // nodes must be genuinely new), the job is admitted purely on real
        // cap headroom (1 real + 2 charge = 3 <= cap 3), `preferred_nodes`
        // stays empty, and real placement lands it on n2+n3.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 128000);
        register_node(&cm, "n2", 8, 128000);
        register_node(&cm, "n3", 8, 128000);
        if let Some(node) = cm.nodes.write().get_mut("n1") {
            node.state = spur_core::node::NodeState::Down;
        }

        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 3);
        cm.qos_cache().insert(Qos {
            name: "cvs".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        {
            let mut jobs = cm.jobs.write();
            let mut r = make_running_job(501, &["n1"], 1);
            r.spec.num_nodes = 1;
            r.spec.qos = Some("cvs".into());
            jobs.insert(501, r);
        }

        let mut newjob = basic_spec("needs-two");
        newjob.qos = Some("cvs".into());
        newjob.num_nodes = 2;
        newjob.num_tasks = 2;
        let new_id = submit_and_wait(&cm, newjob);

        let pending = cm.pending_jobs();
        let job = pending
            .iter()
            .find(|j| j.job_id == new_id)
            .expect("admitted: 1 real (n1) + 2 genuinely new nodes fits the cap of 3");
        assert!(
            job.preferred_nodes.is_empty(),
            "the Down node n1 must not be credited as reusable: got preferred_nodes={:?}",
            job.preferred_nodes
        );

        let nodes = cm.get_nodes();
        let partitions = cm.get_partitions();
        let reservations = cm.get_reservations();
        let cluster_state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &partitions,
            reservations: &reservations,
            topology: None,
        };
        use spur_sched::traits::Scheduler as _;
        let mut scheduler = spur_sched::backfill::BackfillScheduler::new(10);
        let assignments = scheduler.schedule(std::slice::from_ref(job), &cluster_state);
        let assignment = assignments
            .iter()
            .find(|a| a.job_id == new_id)
            .expect("the job must be schedulable on the two healthy idle nodes n2/n3");
        assert!(
            assignment.nodes.iter().all(|n| n != "n1"),
            "must never place on the Down node n1: {:?}",
            assignment.nodes
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn new_distinct_nodes_needed_does_not_credit_a_k0s_reserved_node() {
        // Same gap as the Down-node case, but for a node claimed by the
        // managed k0s cluster: `new_distinct_nodes_needed` must also check
        // `Node::is_k0s_reserved()`, matching the real scheduler's
        // `NodePlacement::matches()` (k0s-reserved nodes are owned by the k8s
        // scheduler and Spur must never place jobs on them). n1 hosts a
        // running QOS job with spare capacity on paper but is claimed for
        // k0s; n2/n3 are healthy and free.
        use spur_core::k0s::K0sRole;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 128000);
        register_node(&cm, "n2", 8, 128000);
        register_node(&cm, "n3", 8, 128000);
        if let Some(node) = cm.nodes.write().get_mut("n1") {
            node.k0s_role = Some(K0sRole::Worker);
        }

        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 3);
        cm.qos_cache().insert(Qos {
            name: "cvs".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        {
            let mut jobs = cm.jobs.write();
            let mut r = make_running_job(502, &["n1"], 1);
            r.spec.num_nodes = 1;
            r.spec.qos = Some("cvs".into());
            jobs.insert(502, r);
        }

        let mut newjob = basic_spec("needs-two-no-k0s");
        newjob.qos = Some("cvs".into());
        newjob.num_nodes = 2;
        newjob.num_tasks = 2;
        let new_id = submit_and_wait(&cm, newjob);

        let pending = cm.pending_jobs();
        let job = pending
            .iter()
            .find(|j| j.job_id == new_id)
            .expect("admitted: 1 real (n1) + 2 genuinely new nodes fits the cap of 3");
        assert!(
            job.preferred_nodes.is_empty(),
            "the k0s-reserved node n1 must not be credited as reusable: got preferred_nodes={:?}",
            job.preferred_nodes
        );

        let nodes = cm.get_nodes();
        let partitions = cm.get_partitions();
        let reservations = cm.get_reservations();
        let cluster_state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &partitions,
            reservations: &reservations,
            topology: None,
        };
        use spur_sched::traits::Scheduler as _;
        let mut scheduler = spur_sched::backfill::BackfillScheduler::new(10);
        let assignments = scheduler.schedule(std::slice::from_ref(job), &cluster_state);
        let assignment = assignments
            .iter()
            .find(|a| a.job_id == new_id)
            .expect("the job must be schedulable on the two healthy idle nodes n2/n3");
        assert!(
            assignment.nodes.iter().all(|n| n != "n1"),
            "must never place on the k0s-reserved node n1: {:?}",
            assignment.nodes
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn in_pass_account_grp_node_block_tags_reason_not_none() {
        // Account grp_tres node=1: the second job is blocked only by the first's
        // in-pass reservation. Expect AssocGrpNodeLimit, not None.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 1);
        cm.association_cache().insert_limits(
            "testuser",
            "research",
            spur_core::accounting::AccountLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
        );

        let mut s1 = basic_spec("a1");
        s1.account = Some("research".into());
        let j1 = submit_and_wait(&cm, s1);
        let mut s2 = basic_spec("a2");
        s2.account = Some("research".into());
        let j2 = submit_and_wait(&cm, s2);

        cm.refresh_pending_reasons();
        assert_eq!(cm.get_job(j1).unwrap().pending_reason, PendingReason::None);
        assert_eq!(
            cm.get_job(j2).unwrap().pending_reason,
            PendingReason::AssocGrpNodeLimit
        );
    }

    #[test]
    fn sum_running_tres_counts_distinct_nodes_not_summed_num_nodes() {
        // Two running jobs share node n2: summed num_nodes = 4, distinct union = 3.
        let mut a = make_running_job(1, &["n1", "n2"], 1);
        a.spec.num_nodes = 2;
        let mut b = make_running_job(2, &["n2", "n3"], 1);
        b.spec.num_nodes = 2;
        let mut jobs = HashMap::new();
        jobs.insert(1, a);
        jobs.insert(2, b);

        let tres = sum_running_tres(&jobs, |_| true);
        assert_eq!(
            tres.get(TresType::Node),
            3,
            "Node TRES must be distinct nodes {{n1,n2,n3}}=3, not summed num_nodes=4"
        );
    }

    #[test]
    fn sum_running_tres_falls_back_to_num_nodes_when_placement_missing() {
        // An unplaced job (num_nodes=3, no recorded placement) must not under-count via the
        // distinct-node union, so it still contributes its full requested count; the two placed
        // jobs share n2, so their contribution is deduped to {n1,n2,n3}=3 rather than summed 2+2=4.
        let mut a = make_running_job(1, &[], 1);
        a.spec.num_nodes = 3;
        a.allocated_nodes.clear();
        let mut b = make_running_job(2, &["n1", "n2"], 1);
        b.spec.num_nodes = 2;
        let mut c = make_running_job(3, &["n2", "n3"], 1);
        c.spec.num_nodes = 2;
        let mut jobs = HashMap::new();
        jobs.insert(1, a);
        jobs.insert(2, b);
        jobs.insert(3, c);

        let tres = sum_running_tres(&jobs, |_| true);
        // Unplaced fallback 3 + placed distinct {n1,n2,n3}=3 = 6, not summed 3+2+2=7.
        assert_eq!(tres.get(TresType::Node), 6);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qos_grp_node_uses_distinct_allocated_nodes_not_summed_num_nodes() {
        // QOS grp node=4, running jobs pack onto a shared node (distinct=3, summed=4); a new
        // 1-node job must pack into the real headroom (3+1=4), not be blocked by the phantom 4+1=5.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        register_node(&cm, "n2", 64, 128000);
        register_node(&cm, "n3", 64, 128000);

        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 4);
        cm.qos_cache().insert(Qos {
            name: "burst".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        // Two running QOS jobs sharing n2: A={n1,n2}, B={n2,n3} -> distinct=3,
        // summed num_nodes=4.
        {
            let mut jobs = cm.jobs.write();
            for (id, nodes) in [(101u32, ["n1", "n2"]), (102, ["n2", "n3"])] {
                let mut j = make_running_job(id, &nodes, 1);
                j.spec.num_nodes = 2;
                j.spec.qos = Some("burst".into());
                jobs.insert(id, j);
            }
        }

        let mut newjob = basic_spec("pack-me");
        newjob.qos = Some("burst".into());
        newjob.num_nodes = 1;
        let new_id = submit_and_wait(&cm, newjob);

        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        assert!(
            pending.contains(&new_id),
            "1-node job must pack into distinct-node headroom (3+1=4 <= 4); \
             summing num_nodes (4+1=5) wrongly blocks it on QOSGrpNodeLimit"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qos_grp_node_still_blocks_when_distinct_cap_reached() {
        // The distinct-node fix must not over-admit: with grp node=3 already fully
        // occupied by distinct nodes {n1,n2,n3}, a job needing a 4th distinct node
        // (num_nodes=4) is still blocked on QOSGrpNodeLimit.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        for n in ["n1", "n2", "n3", "n4"] {
            register_node(&cm, n, 64, 128000);
        }

        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 3);
        cm.qos_cache().insert(Qos {
            name: "burst".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        {
            let mut jobs = cm.jobs.write();
            for (id, nodes) in [(101u32, ["n1", "n2"]), (102, ["n2", "n3"])] {
                let mut j = make_running_job(id, &nodes, 1);
                j.spec.num_nodes = 2;
                j.spec.qos = Some("burst".into());
                jobs.insert(id, j);
            }
        }

        let mut newjob = basic_spec("too-big");
        newjob.qos = Some("burst".into());
        newjob.num_nodes = 4;
        newjob.num_tasks = 4; // avoid effective_num_nodes() clamping to num_tasks
        let new_id = submit_and_wait(&cm, newjob);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(new_id).unwrap().pending_reason,
            PendingReason::QosGrpNodeLimit,
            "distinct occupancy 3 + requested 4 = 7 > grp cap 3 must still block"
        );
        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        assert!(!pending.contains(&new_id));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn in_pass_bb_exhaustion_tags_reason_not_none() {
        // Two jobs want 60GB from a 100GB BB pool: the second, blocked only by
        // the first's in-pass reservation, must surface BurstBufferResources.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        *cm.burst_buffer_total_gb.write() = 100;

        let mut s1 = basic_spec("bb1");
        s1.burst_buffer = Some("capacity=60".into());
        let _j1 = submit_and_wait(&cm, s1);
        let mut s2 = basic_spec("bb2");
        s2.burst_buffer = Some("capacity=60".into());
        let j2 = submit_and_wait(&cm, s2);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(j2).unwrap().pending_reason,
            PendingReason::BurstBufferResources
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn account_limits_do_not_block_jobs_without_an_account() {
        // A job with no account can't be constrained by an association it
        // doesn't belong to.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.association_cache().insert_limits(
            "testuser",
            "research",
            spur_core::accounting::AccountLimits {
                max_running_jobs: Some(0),
                ..Default::default()
            },
        );

        let spec = basic_spec("d1");
        let job_id = submit_and_wait(&cm, spec);

        cm.refresh_pending_reasons();
        assert_ne!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::AssocMaxJobsLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn running_job_license_consumption_blocks_next_job() {
        // Concurrent license accounting: a running job holding all of a license
        // must make a second job requesting that license ineligible, even though
        // each request alone is within the configured total.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        cm.license_pool.write().insert("fluent".into(), 2);

        let mut s1 = basic_spec("j1");
        s1.gres = vec!["license:fluent:2".into()];
        let j1 = submit_and_wait(&cm, s1);
        let res = scalar_alloc(1, 1000);
        cm.start_job(
            j1,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, j1, JobState::Running);

        assert_eq!(
            cm.available_licenses().get("fluent").copied(),
            Some(0),
            "running job's licenses should count as in use (none available)"
        );

        let mut s2 = basic_spec("j2");
        s2.gres = vec!["license:fluent:1".into()];
        let j2 = submit_and_wait(&cm, s2);
        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        assert!(
            !pending.contains(&j2),
            "j2 must be blocked while the license pool is exhausted"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn terminal_job_frees_its_licenses_without_drifting_total() {
        // Derived accounting: a job releases its licenses the moment it leaves the
        // active set, and the configured total is never mutated (no drift).
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        cm.license_pool.write().insert("fluent".into(), 2);

        let mut s = basic_spec("j");
        s.gres = vec!["license:fluent:2".into()];
        let id = submit_and_wait(&cm, s);
        let res = scalar_alloc(1, 1000);
        cm.start_job(
            id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);
        assert_eq!(cm.available_licenses().get("fluent").copied(), Some(0));

        cm.cancel_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Cancelled);
        assert_eq!(
            cm.available_licenses().get("fluent").copied(),
            Some(2),
            "licenses must be freed when the job leaves the active set"
        );
        assert_eq!(
            *cm.license_pool.read().get("fluent").unwrap(),
            2,
            "configured total must never be mutated"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scheduling_pass_stamps_last_sched_eval() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        let job_id = cm.submit_job(basic_spec("last-eval")).unwrap().job_id;
        wait_for("job visible", || cm.get_job(job_id).is_some());
        assert!(
            cm.get_job(job_id).unwrap().last_sched_eval.is_none(),
            "a job must not claim to have been evaluated before any cycle ran"
        );

        cm.pending_jobs_and_tag_reasons();

        let stamped = cm.get_job(job_id).unwrap().last_sched_eval;
        assert!(
            stamped.is_some(),
            "scheduling pass must record LastSchedEval"
        );

        cm.pending_jobs_and_tag_reasons();
        let restamped = cm.get_job(job_id).unwrap().last_sched_eval;
        assert!(
            restamped > stamped,
            "each cycle must advance LastSchedEval, got {stamped:?} then {restamped:?}"
        );

        // A job that started between the read snapshot and the write lock must
        // not gain a stamp — the re-check inside mark_scheduler_evaluated.
        cm.jobs.write().get_mut(&job_id).unwrap().state = JobState::Running;
        cm.mark_scheduler_evaluated(&[job_id]);
        assert_eq!(
            cm.get_job(job_id).unwrap().last_sched_eval,
            restamped,
            "a running job must not be stamped as scheduler-evaluated"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn array_throttle_reserves_slots_within_classification() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        let mut spec = basic_spec("arr-throttle-pass");
        spec.array_spec = Some("0-2%1".into());
        let parent_id = cm.submit_job(spec).unwrap().job_id;
        let task_ids: Vec<JobId> = (1..=3).map(|offset| parent_id + offset).collect();
        for id in &task_ids {
            wait_for(&format!("array task {id}"), || cm.get_job(*id).is_some());
        }

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|job| job.job_id)
            .collect();
        let admitted = task_ids.iter().filter(|id| pending.contains(id)).count();
        assert_eq!(admitted, 1);
        for id in task_ids.iter().filter(|id| !pending.contains(id)) {
            assert_eq!(
                cm.get_job(*id).unwrap().pending_reason,
                PendingReason::JobArrayTaskLimit
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn array_throttle_excludes_task_from_pending_classification() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        let mut spec = basic_spec("arr-throttle");
        spec.array_spec = Some("0-2%1".into());
        let parent_id = cm.submit_job(spec).unwrap().job_id;
        let task_ids: Vec<JobId> = (1..=3).map(|offset| parent_id + offset).collect();
        for id in &task_ids {
            wait_for(&format!("array task {id}"), || cm.get_job(*id).is_some());
        }

        start_job_on(&cm, task_ids[0], "n1");
        settle(&cm, task_ids[0], JobState::Running);

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|j| j.job_id)
            .collect();
        for id in &task_ids[1..] {
            assert!(
                !pending.contains(id),
                "array-throttled task {id} must be excluded from scheduling"
            );
            assert_eq!(
                cm.get_job(*id).unwrap().pending_reason,
                PendingReason::JobArrayTaskLimit
            );
        }

        cm.cancel_job(task_ids[0], "testuser").unwrap();
        settle(&cm, task_ids[0], JobState::Cancelled);

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|job| job.job_id)
            .collect();
        let admitted: Vec<JobId> = task_ids[1..]
            .iter()
            .copied()
            .filter(|id| pending.contains(id))
            .collect();
        assert_eq!(admitted.len(), 1);
        assert_eq!(
            cm.get_job(admitted[0]).unwrap().pending_reason,
            PendingReason::None
        );

        let still_throttled = task_ids[1..]
            .iter()
            .copied()
            .find(|id| *id != admitted[0])
            .unwrap();
        assert_eq!(
            cm.get_job(still_throttled).unwrap().pending_reason,
            PendingReason::JobArrayTaskLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn array_throttle_does_not_consume_scarce_license_pool() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        cm.license_pool.write().insert("fluent".into(), 2);

        let mut spec = basic_spec("arr-lic");
        spec.array_spec = Some("0-1%1".into());
        spec.gres = vec!["license:fluent:1".into()];
        let parent_id = cm.submit_job(spec).unwrap().job_id;
        let t1 = parent_id + 1;
        let t2 = parent_id + 2;
        wait_for("array license tasks", || {
            cm.get_job(t1).is_some() && cm.get_job(t2).is_some()
        });

        let res = scalar_alloc(1, 1000);
        cm.start_job(
            t1,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, t1, JobState::Running);

        let solo = submit_and_wait(&cm, {
            let mut s = basic_spec("solo-lic");
            s.gres = vec!["license:fluent:1".into()];
            s
        });

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|j| j.job_id)
            .collect();
        assert!(
            !pending.contains(&t2),
            "array-throttled task must not enter the scheduling set"
        );
        assert!(
            pending.contains(&solo),
            "the eligible job must still receive the remaining license slot"
        );
        assert_eq!(
            cm.get_job(t2).unwrap().pending_reason,
            PendingReason::JobArrayTaskLimit
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn array_throttle_reason_hands_off_to_license_blocker() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.license_pool.write().insert("fluent".into(), 1);

        let mut spec = basic_spec("arr-throttle-license");
        spec.array_spec = Some("0-1%1".into());
        spec.gres = vec!["license:fluent:2".into()];
        let parent_id = cm.submit_job(spec).unwrap().job_id;
        let t1 = parent_id + 1;
        let t2 = parent_id + 2;
        wait_for("array license tasks", || {
            cm.get_job(t1).is_some() && cm.get_job(t2).is_some()
        });

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|job| job.job_id)
            .collect();

        assert!(!pending.contains(&t1));
        assert!(!pending.contains(&t2));
        assert_eq!(
            cm.get_job(t1).unwrap().pending_reason,
            PendingReason::Licenses
        );
        assert_eq!(
            license_block(&cm.get_job(t2).unwrap(), &cm.available_licenses()),
            Some(PendingReason::Licenses)
        );
        assert_eq!(
            cm.get_job(t2).unwrap().pending_reason,
            PendingReason::JobArrayTaskLimit
        );

        cm.cancel_job(t1, "testuser").unwrap();
        settle(&cm, t1, JobState::Cancelled);

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|job| job.job_id)
            .collect();

        assert!(!pending.contains(&t2));
        assert_eq!(
            cm.get_job(t2).unwrap().pending_reason,
            PendingReason::Licenses
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn array_throttle_does_not_consume_scarce_bb_pool() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        *cm.burst_buffer_total_gb.write() = 160;

        let mut spec = basic_spec("arr-bb");
        spec.array_spec = Some("0-1%1".into());
        spec.burst_buffer = Some("capacity=60".into());
        let parent_id = cm.submit_job(spec).unwrap().job_id;
        let t1 = parent_id + 1;
        let t2 = parent_id + 2;
        wait_for("array bb tasks", || {
            cm.get_job(t1).is_some() && cm.get_job(t2).is_some()
        });

        start_job_on(&cm, t1, "n1");
        settle(&cm, t1, JobState::Running);

        let solo = submit_and_wait(&cm, {
            let mut s = basic_spec("solo-bb");
            s.burst_buffer = Some("capacity=60".into());
            s
        });

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|j| j.job_id)
            .collect();
        assert!(
            !pending.contains(&t2),
            "array-throttled task must not enter the scheduling set"
        );
        assert_eq!(
            cm.available_bb(),
            40,
            "only the running task and eligible solo job may reserve capacity"
        );
        assert_eq!(cm.get_job(t2).unwrap().bb_stage_state, BbStageState::None);
        assert_eq!(
            cm.get_job(t2).unwrap().pending_reason,
            PendingReason::JobArrayTaskLimit
        );
        assert_eq!(
            cm.get_job(solo).unwrap().bb_stage_state,
            BbStageState::Staging
        );

        assert!(cm.complete_bb_stage_in(solo));
        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|j| j.job_id)
            .collect();
        assert!(pending.contains(&solo));
        assert!(!pending.contains(&t2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn future_begin_jobs_do_not_receive_consumable_block_reasons() {
        // A consumable shortage is transient and may well be gone by the time a
        // deferred job's start time arrives, so the hold is the honest reason to
        // report and the shortage must not be recorded in its place.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        *cm.burst_buffer_total_gb.write() = 100;
        let begin_time = Utc::now() + chrono::Duration::hours(1);

        let mut license_spec = basic_spec("future-license");
        license_spec.begin_time = Some(begin_time);
        license_spec.gres = vec!["license:missing:1".into()];
        let license_id = submit_and_wait(&cm, license_spec);

        let mut bb_spec = basic_spec("future-bb");
        bb_spec.begin_time = Some(begin_time);
        bb_spec.burst_buffer = Some("capacity=500".into());
        let bb_id = submit_and_wait(&cm, bb_spec);

        // Seed an unrelated reason so the pass has something to overwrite: the
        // hold has to be recorded even when the job already carries a reason.
        {
            let mut jobs = cm.jobs.write();
            jobs.get_mut(&license_id).unwrap().pending_reason = PendingReason::Priority;
            jobs.get_mut(&bb_id).unwrap().pending_reason = PendingReason::Priority;
        }

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|job| job.job_id)
            .collect();

        assert!(!pending.contains(&license_id));
        assert!(!pending.contains(&bb_id));
        assert_eq!(
            cm.get_job(license_id).unwrap().pending_reason,
            PendingReason::BeginTime,
            "a missing license must not displace the deferral as the reason"
        );
        assert_eq!(
            cm.get_job(bb_id).unwrap().pending_reason,
            PendingReason::BeginTime,
            "a burst-buffer shortfall must not displace the deferral as the reason"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_does_not_overallocate_licenses_within_one_pass() {
        // Two jobs each request fluent:1 from a pool of 1: the loser waits on
        // Licenses (not None), the same as an outright shortage.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        cm.license_pool.write().insert("fluent".into(), 1);

        let mut s1 = basic_spec("a");
        s1.gres = vec!["license:fluent:1".into()];
        let a = submit_and_wait(&cm, s1);
        let mut s2 = basic_spec("b");
        s2.gres = vec!["license:fluent:1".into()];
        let b = submit_and_wait(&cm, s2);

        let pending: Vec<JobId> = cm
            .pending_jobs_and_tag_reasons()
            .iter()
            .map(|j| j.job_id)
            .collect();
        let granted = [a, b].iter().filter(|id| pending.contains(id)).count();
        assert_eq!(
            granted, 1,
            "pending_jobs() returned {granted} fluent jobs but the pool holds only 1"
        );
        let loser = [a, b].into_iter().find(|id| !pending.contains(id)).unwrap();
        assert_eq!(
            cm.get_job(loser).unwrap().pending_reason,
            PendingReason::Licenses,
            "the job that lost the single license must wait on Licenses, not None"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_does_not_oversubscribe_qos_grp_node_within_one_pass() {
        // Three pending 1-node jobs share a QOS capped at grp_tres node=2. Each
        // fits against the empty running total, but a single pass must admit at
        // most 2 or the QOSGrpNodeLimit is over-subscribed.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 2);
        cm.qos_cache().insert(Qos {
            name: "burst".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        let ids: Vec<JobId> = (0..3)
            .map(|i| {
                let mut s = basic_spec(&format!("b{i}"));
                s.qos = Some("burst".into());
                submit_and_wait(&cm, s)
            })
            .collect();

        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        let granted = ids.iter().filter(|id| pending.contains(id)).count();
        assert_eq!(
            granted, 2,
            "pending_jobs() returned {granted} burst jobs but grp_tres node=2 allows 2"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_qos_grp_reserves_highest_priority_first() {
        // A high-priority 2-node job and a low-priority 1-node job share a QOS
        // capped at grp_tres node=2. The pass must keep the high-priority job
        // (which alone fills the cap) and drop the low-priority one, not admit
        // the small job first and starve the large one.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 2);
        cm.qos_cache().insert(Qos {
            name: "burst".into(),
            limits: spur_core::accounting::QosLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
            ..Default::default()
        });

        let mut big = basic_spec("big");
        big.qos = Some("burst".into());
        big.num_nodes = 2;
        big.num_tasks = 2;
        big.priority = Some(1000);
        let big_id = submit_and_wait(&cm, big);

        let mut small = basic_spec("small");
        small.qos = Some("burst".into());
        small.num_nodes = 1;
        small.priority = Some(1);
        let small_id = submit_and_wait(&cm, small);

        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        assert!(
            pending.contains(&big_id),
            "the highest-priority job must win the group reservation"
        );
        assert!(
            !pending.contains(&small_id),
            "the low-priority job must not consume headroom the big job needs"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_does_not_oversubscribe_account_grp_node_within_one_pass() {
        // Same one-pass over-subscription guard, one layer up: an account
        // association capped at grp_tres node=2 must not admit 3 pending 1-node
        // jobs in a single pass.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        let mut grp = TresRecord::new();
        grp.set(TresType::Node, 2);
        cm.association_cache().insert_limits(
            "testuser",
            "research",
            spur_core::accounting::AccountLimits {
                grp_tres: Some(grp),
                ..Default::default()
            },
        );

        let ids: Vec<JobId> = (0..3)
            .map(|i| {
                let mut s = basic_spec(&format!("a{i}"));
                s.account = Some("research".into());
                submit_and_wait(&cm, s)
            })
            .collect();

        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        let granted = ids.iter().filter(|id| pending.contains(id)).count();
        assert_eq!(
            granted, 2,
            "pending_jobs() returned {granted} account jobs but grp_tres node=2 allows 2"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_does_not_oversubscribe_account_max_running_jobs_within_one_pass() {
        // Same guard as the QOS max_jobs_per_user test, one layer up: an account
        // association capped at max_running_jobs=2 must not admit 3 pending jobs
        // from one user in a single pass even though none is running yet.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        cm.association_cache().insert_limits(
            "testuser",
            "research",
            spur_core::accounting::AccountLimits {
                max_running_jobs: Some(2),
                ..Default::default()
            },
        );

        let ids: Vec<JobId> = (0..3)
            .map(|i| {
                let mut s = basic_spec(&format!("m{i}"));
                s.account = Some("research".into());
                submit_and_wait(&cm, s)
            })
            .collect();

        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        let granted = ids.iter().filter(|id| pending.contains(id)).count();
        assert_eq!(
            granted, 2,
            "pending_jobs() returned {granted} account jobs but max_running_jobs=2 allows 2"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_does_not_oversubscribe_qos_max_jobs_per_user_within_one_pass() {
        // The per-user running-job count cap is a same-pass accumulator too: three
        // pending jobs from one user under a QOS capped at max_jobs_per_user=2
        // must not all be admitted in a single pass even though none is running yet.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        cm.qos_cache().insert(Qos {
            name: "cnt".into(),
            limits: spur_core::accounting::QosLimits {
                max_jobs_per_user: Some(2),
                ..Default::default()
            },
            ..Default::default()
        });

        let ids: Vec<JobId> = (0..3)
            .map(|i| {
                let mut s = basic_spec(&format!("c{i}"));
                s.qos = Some("cnt".into());
                submit_and_wait(&cm, s)
            })
            .collect();

        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        let granted = ids.iter().filter(|id| pending.contains(id)).count();
        assert_eq!(
            granted, 2,
            "pending_jobs() returned {granted} jobs but max_jobs_per_user=2 allows 2"
        );
    }

    /// A cluster with accounting configured, ready to be given a cold cache.
    async fn accounting_cluster(dir: &TempDir) -> Arc<ClusterManager> {
        let mut cfg = test_config();
        cfg.accounting.database_url = "postgres://unused-in-test".into();
        test_cluster_with_config(dir, cfg).await
    }

    /// Three jobs from one user under a QOS capped at two, returned in submit order.
    fn submit_three_under_capped_qos(cm: &ClusterManager) -> Vec<JobId> {
        cm.qos_cache().insert(Qos {
            name: "cnt".into(),
            limits: spur_core::accounting::QosLimits {
                max_jobs_per_user: Some(2),
                ..Default::default()
            },
            ..Default::default()
        });
        (0..3)
            .map(|i| {
                let mut s = basic_spec(&format!("c{i}"));
                s.qos = Some("cnt".into());
                submit_and_wait(cm, s)
            })
            .collect()
    }

    fn admitted_of(cm: &ClusterManager, ids: &[JobId]) -> usize {
        let admitted: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        ids.iter().filter(|id| admitted.contains(id)).count()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_keeps_qos_limits_across_a_controller_restart() {
        // Submit accepts a `--qos` only while the cache knows the name, so a queued
        // job always carries a real QOS. A restarting controller starts with an
        // empty cache and its predecessor's queue already durable, and that queue
        // must not be admitted past the QOS's caps just because the limits are not
        // readable yet.
        let dir = TempDir::new().unwrap();
        let cm = accounting_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        let ids = submit_three_under_capped_qos(&cm);

        cm.qos_cache().reset();

        let granted = admitted_of(&cm, &ids);
        assert_eq!(
            granted, 0,
            "pending_jobs() admitted {granted} of 3 jobs under max_jobs_per_user=2 \
             while the QOS cache was unreadable"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_reports_a_cold_cache_as_the_hold_reason() {
        // The hold has to be legible: an operator seeing these jobs pend needs the
        // cold cache named, not a generic Resources or Priority wait.
        let dir = TempDir::new().unwrap();
        let cm = accounting_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        let ids = submit_three_under_capped_qos(&cm);

        cm.qos_cache().reset();
        cm.refresh_pending_reasons();

        assert_eq!(
            cm.get_job(ids[0]).unwrap().pending_reason,
            PendingReason::AccountingUnavailable
        );
    }

    // A job waiting on its start time is not waiting on the cache: the cold-cache
    // hold gates admission, and a begin-deferred job is not up for admission, so
    // BeginTime stays its reason.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cold_cache_does_not_claim_a_begin_deferred_job() {
        let dir = TempDir::new().unwrap();
        let cm = accounting_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        cm.qos_cache().insert(Qos {
            name: "cnt".into(),
            ..Default::default()
        });

        let mut spec = basic_spec("later");
        spec.qos = Some("cnt".into());
        spec.begin_time = Some(Utc::now() + chrono::Duration::hours(1));
        let id = submit_and_wait(&cm, spec);

        cm.qos_cache().reset();
        cm.refresh_pending_reasons();

        assert_eq!(
            cm.get_job(id).unwrap().pending_reason,
            PendingReason::BeginTime
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_holds_a_job_whose_qos_was_deleted() {
        // A loaded cache that does not know the name is a QOS removed after the job
        // queued, not an unreadable one — the same limitless-default hazard, and the
        // case Slurm reports as InvalidQOS.
        let dir = TempDir::new().unwrap();
        let cm = accounting_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        let ids = submit_three_under_capped_qos(&cm);

        // A later refresh that no longer carries "cnt": loaded, but without it.
        cm.qos_cache().reset();
        cm.qos_cache().insert(Qos {
            name: "other".into(),
            ..Default::default()
        });
        cm.refresh_pending_reasons();

        assert_eq!(admitted_of(&cm, &ids), 0);
        assert_eq!(
            cm.get_job(ids[0]).unwrap().pending_reason,
            PendingReason::InvalidQOS
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_holds_account_jobs_while_the_association_cache_is_cold() {
        // The association cache carries the same caps one layer up and starts just
        // as empty, so an account-scoped job gets the same hold.
        let dir = TempDir::new().unwrap();
        let cm = accounting_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        let mut spec = basic_spec("acct");
        spec.account = Some("tenant-a".into());
        // Submit has its own cold-cache gate, so the association is seeded to get
        // the job queued; the scheduling pass then faces a cache that went cold.
        cm.association_cache()
            .insert_association(&spec.user, "tenant-a");
        let id = submit_and_wait(&cm, spec);
        cm.association_cache().reset();

        assert_eq!(admitted_of(&cm, &[id]), 0);
        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(id).unwrap().pending_reason,
            PendingReason::AccountingUnavailable
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_admits_freely_when_accounting_is_disabled() {
        // Without accounting there are no limits to read, so an empty cache means
        // "no caps exist" rather than "caps are unreadable" and must not hold work.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        let ids = submit_three_under_capped_qos(&cm);

        cm.qos_cache().reset();

        assert_eq!(
            admitted_of(&cm, &ids),
            3,
            "a cold cache must not gate scheduling on a cluster that runs no accounting"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_still_enforces_the_cap_once_the_cache_is_warm() {
        // The guard must not swallow the limit it protects: with the cache loaded
        // the cap applies exactly as before.
        let dir = TempDir::new().unwrap();
        let cm = accounting_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        let ids = submit_three_under_capped_qos(&cm);

        assert_eq!(admitted_of(&cm, &ids), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cold_cache_hold_clears_once_the_qos_cache_reloads() {
        // The fail-closed hold must be self-clearing. A controller that boots
        // with the database down keeps the cache cold and holds every job; once
        // the background bring-up connects and a refresh lands, the same
        // scheduling pass that held these jobs must admit them — no operator
        // release. This mirrors that transition at the cache boundary the
        // scheduler reads.
        let dir = TempDir::new().unwrap();
        let cm = accounting_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        let ids = submit_three_under_capped_qos(&cm);

        cm.qos_cache().reset();
        assert_eq!(admitted_of(&cm, &ids), 0, "a cold cache holds every job");
        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(ids[0]).unwrap().pending_reason,
            PendingReason::AccountingUnavailable
        );

        cm.qos_cache().insert(Qos {
            name: "cnt".into(),
            limits: spur_core::accounting::QosLimits {
                max_jobs_per_user: Some(2),
                ..Default::default()
            },
            ..Default::default()
        });

        assert_eq!(
            admitted_of(&cm, &ids),
            2,
            "once the cache reloads the hold clears and the cap applies"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn limits_readable_tracks_enabled_and_per_cache_state() {
        // The canonical "can limits be read?" verdict is per cache: a cold cache
        // on an accounting cluster is unreadable, and a warm QOS cache does not
        // make the still-cold association cache readable. This split is exactly
        // what a conjoined loaded-only check gets wrong.
        let dir = TempDir::new().unwrap();
        let cm = accounting_cluster(&dir).await;
        assert!(!cm.qos_limits_readable());
        assert!(!cm.association_limits_readable());

        cm.qos_cache().insert(Qos {
            name: "any".into(),
            ..Default::default()
        });
        assert!(cm.qos_limits_readable());
        assert!(
            !cm.association_limits_readable(),
            "a warm QOS cache must not vouch for a cold association cache"
        );

        cm.association_cache()
            .insert_association("alice", "research");
        assert!(cm.association_limits_readable());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn limits_readable_is_true_when_accounting_is_disabled() {
        // With no database there are no caps to read, so both caches read as
        // "readable" even cold — the accounting-disabled case a loaded-only
        // check misreports as "unreadable" forever.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        assert!(cm.qos_limits_readable());
        assert!(cm.association_limits_readable());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_jobs_does_not_oversubscribe_qos_max_tres_per_user_within_one_pass() {
        // The per-user TRES cap (distinct from the group cap) also accumulates
        // within a pass: two 3-cpu jobs from one user under maxtresperuser cpu=4
        // must not both be admitted (3+3 > 4), even with an empty running set.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 64, 128000);
        let mut per_user = TresRecord::new();
        per_user.set(TresType::Cpu, 4);
        cm.qos_cache().insert(Qos {
            name: "usertres".into(),
            limits: spur_core::accounting::QosLimits {
                max_tres_per_user: Some(per_user),
                ..Default::default()
            },
            ..Default::default()
        });

        let ids: Vec<JobId> = (0..2)
            .map(|i| {
                let mut s = basic_spec(&format!("u{i}"));
                s.qos = Some("usertres".into());
                s.num_tasks = 3;
                submit_and_wait(&cm, s)
            })
            .collect();

        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        let granted = ids.iter().filter(|id| pending.contains(id)).count();
        assert_eq!(
            granted, 1,
            "pending_jobs() returned {granted} jobs but maxtresperuser cpu=4 fits only one 3-cpu job"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bb_request_over_pool_sets_resources_reason() {
        // A job asking for more BB capacity than the pool holds stays PENDING
        // with BurstBufferResources, and pending_jobs() drops it from scheduling.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        *cm.burst_buffer_total_gb.write() = 100;

        let mut spec = basic_spec("bb-too-big");
        spec.burst_buffer = Some("capacity=500".into());
        let job_id = submit_and_wait(&cm, spec);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::BurstBufferResources
        );
        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        assert!(
            !pending.contains(&job_id),
            "a job over the BB pool must be dropped from pending_jobs()"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bb_stage_in_holds_then_becomes_dispatchable() {
        // A BB job that fits the pool reserves capacity (None -> Staging), is
        // held with BurstBufferStageIn and excluded from dispatch, then becomes
        // dispatchable once stage-in completes (Staging -> Ready).
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        *cm.burst_buffer_total_gb.write() = 100;

        let mut spec = basic_spec("bb-stage");
        spec.burst_buffer = Some("capacity=40".into());
        let job_id = submit_and_wait(&cm, spec);

        cm.advance_bb_staging();
        assert_eq!(
            cm.get_job(job_id).unwrap().bb_stage_state,
            BbStageState::Staging
        );
        assert_eq!(cm.available_bb(), 60);

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::BurstBufferStageIn
        );
        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        assert!(
            !pending.contains(&job_id),
            "a staging BB job must not be dispatched until stage-in completes"
        );

        assert!(cm.complete_bb_stage_in(job_id));
        assert_eq!(
            cm.get_job(job_id).unwrap().bb_stage_state,
            BbStageState::Ready
        );
        let pending: Vec<JobId> = cm.pending_jobs().iter().map(|j| j.job_id).collect();
        assert!(
            pending.contains(&job_id),
            "a Ready BB job must be dispatchable"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bb_staging_does_not_oversubscribe_pool() {
        // Two BB jobs each want 60GB but the pool holds 100. advance_bb_staging()
        // must reserve for only one; the other stays None and is reported as a
        // resource shortage.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        *cm.burst_buffer_total_gb.write() = 100;

        let mut s1 = basic_spec("bb-a");
        s1.burst_buffer = Some("capacity=60".into());
        let a = submit_and_wait(&cm, s1);
        let mut s2 = basic_spec("bb-b");
        s2.burst_buffer = Some("capacity=60".into());
        let b = submit_and_wait(&cm, s2);

        let staged = cm.advance_bb_staging();
        assert_eq!(staged.len(), 1, "only one 60GB job fits a 100GB pool");

        let states: Vec<(JobId, BbStageState)> = [a, b]
            .iter()
            .map(|id| (*id, cm.get_job(*id).unwrap().bb_stage_state))
            .collect();
        let staging = states
            .iter()
            .filter(|(_, s)| *s == BbStageState::Staging)
            .count();
        let none = states
            .iter()
            .filter(|(_, s)| *s == BbStageState::None)
            .count();
        assert_eq!((staging, none), (1, 1), "exactly one job stages");

        cm.refresh_pending_reasons();
        let unstaged = states
            .iter()
            .find(|(_, s)| *s == BbStageState::None)
            .map(|(id, _)| *id)
            .unwrap();
        assert_eq!(
            cm.get_job(unstaged).unwrap().pending_reason,
            PendingReason::BurstBufferResources
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bb_capacity_freed_when_job_completes() {
        // A BB job releases its reserved capacity when it leaves the active set,
        // and the configured total is never mutated.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        *cm.burst_buffer_total_gb.write() = 100;

        let mut spec = basic_spec("bb-life");
        spec.burst_buffer = Some("capacity=40".into());
        let id = submit_and_wait(&cm, spec);

        cm.advance_bb_staging();
        assert!(cm.complete_bb_stage_in(id));
        assert_eq!(cm.available_bb(), 60);

        let res = scalar_alloc(1, 1000);
        cm.start_job(
            id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);
        assert_eq!(cm.available_bb(), 60, "running BB job still holds capacity");

        cm.cancel_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Cancelled);
        assert_eq!(
            cm.available_bb(),
            100,
            "capacity must be freed when the job leaves the active set"
        );
        assert_eq!(
            *cm.burst_buffer_total_gb.read(),
            100,
            "configured total must never be mutated"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tag_blocked_preserves_held_reason() {
        // A user-held job blocked by a reservation must stay Held, not get
        // reclassified to Reservation.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let mut spec = basic_spec("held");
        spec.reservation = Some("does-not-exist".into());
        let job_id = submit_and_wait(&cm, spec);
        {
            let mut jobs = cm.jobs.write();
            jobs.get_mut(&job_id).unwrap().pending_reason = PendingReason::Held;
        }

        cm.refresh_pending_reasons();
        assert_eq!(
            cm.get_job(job_id).unwrap().pending_reason,
            PendingReason::Held
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_classification_uses_first_block_reason() {
        // Blocked by both a dependency and an absent reservation -> Dependency
        // wins and the same classification excludes the job from scheduling.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // Parent running -> child's afterok dependency is Waiting (not satisfied).
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("parent")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));

        let mut child = basic_spec("child");
        child.dependency = vec!["afterok:1".into()];
        child.reservation = Some("does-not-exist".into());
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 2,
            spec: Box::new(child),
        });

        let pending = cm.pending_jobs_and_tag_reasons();
        assert!(!pending.iter().any(|job| job.job_id == 2));
        assert_eq!(
            cm.get_job(2).unwrap().pending_reason,
            PendingReason::Dependency
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn complete_terminal_job_errors() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let job_id = submit_and_wait(&cm, basic_spec("j"));
        cm.cancel_job(job_id, "testuser").unwrap();
        settle(&cm, job_id, JobState::Cancelled);

        let result = cm.complete_job(job_id, 1, JobState::Failed);
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_running_job_releases_resources() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "worker1", 8, 16000);
        let job_id = submit_and_wait(&cm, basic_spec("cancel-alloc"));

        let resources = scalar_alloc(2, 4000);
        cm.start_job(
            job_id,
            vec!["worker1".into()],
            resources.clone(),
            per_node_for(&["worker1"], resources),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        let node = cm.get_node("worker1").unwrap();
        assert_eq!(node.alloc_resources.cpus, 2);

        cm.cancel_job(job_id, "testuser").unwrap();
        settle(&cm, job_id, JobState::Cancelled);

        let node = cm.get_node("worker1").unwrap();
        assert_eq!(
            node.alloc_resources.cpus, 0,
            "resources must be freed after cancel"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn double_cancel_returns_error() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let job_id = submit_and_wait(&cm, basic_spec("double-cancel"));
        cm.cancel_job(job_id, "testuser").unwrap();
        settle(&cm, job_id, JobState::Cancelled);

        let result = cm.cancel_job(job_id, "testuser");
        assert!(
            result.is_err(),
            "cancelling an already-cancelled job must fail"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_job_wrong_user_rejected() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let job_id = submit_and_wait(&cm, basic_spec("auth-cancel"));
        let result = cm.cancel_job(job_id, "other_user");
        assert!(
            result.is_err(),
            "non-owner must not cancel another user's job"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot") && err_msg.contains("cancel"),
            "error should mention the denied action: {err_msg}"
        );

        // Job must still be alive.
        let job = cm.get_job(job_id).unwrap();
        assert!(!job.state.is_terminal());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_job_root_allowed() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let job_id = submit_and_wait(&cm, basic_spec("root-cancel"));
        cm.cancel_job(job_id, "root").unwrap();
        settle(&cm, job_id, JobState::Cancelled);
        assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_job_empty_user_allowed() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let job_id = submit_and_wait(&cm, basic_spec("internal-cancel"));
        cm.cancel_job(job_id, "").unwrap();
        settle(&cm, job_id, JobState::Cancelled);
        assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn suspend_job_wrong_user_rejected() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let id = submit_and_wait(&cm, basic_spec("sus-auth"));
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);

        let result = cm.suspend_job(id, "other_user");
        assert!(
            result.is_err(),
            "non-owner must not suspend another user's job"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot") && err_msg.contains("suspend"),
            "error should mention the denied action: {err_msg}"
        );
        assert_eq!(cm.get_job(id).unwrap().state, JobState::Running);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn suspend_job_root_allowed() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let id = submit_and_wait(&cm, basic_spec("sus-root"));
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);

        cm.suspend_job(id, "root").unwrap();
        settle(&cm, id, JobState::Suspended);
        assert_eq!(cm.get_job(id).unwrap().state, JobState::Suspended);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn resume_job_wrong_user_rejected() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let id = submit_and_wait(&cm, basic_spec("res-auth"));
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);
        cm.suspend_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Suspended);

        let result = cm.resume_job(id, "other_user");
        assert!(
            result.is_err(),
            "non-owner must not resume another user's job"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot") && err_msg.contains("resume"),
            "error should mention the denied action: {err_msg}"
        );
        assert_eq!(cm.get_job(id).unwrap().state, JobState::Suspended);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn resume_job_root_allowed() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        let id = submit_and_wait(&cm, basic_spec("res-root"));
        let res = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);
        cm.suspend_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Suspended);

        cm.resume_job(id, "root").unwrap();
        settle(&cm, id, JobState::Running);
        assert_eq!(cm.get_job(id).unwrap().state, JobState::Running);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_and_restore() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);
        submit_and_wait(&cm, basic_spec("snap-job"));

        let data = cm.snapshot_state().unwrap();
        assert!(!data.is_empty());

        // Create a fresh cluster and restore
        let dir2 = TempDir::new().unwrap();
        let cm2 = test_cluster(&dir2).await;
        cm2.restore_from_snapshot(&data).unwrap();

        assert!(cm2.get_job(1).is_some());
        assert!(cm2.get_node("n1").is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn k0s_state_survives_snapshot() {
        use spur_core::k0s::{K0sPhase, K0sRole};
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);
        cm.assign_node_k0s("n1", K0sRole::Worker, "10.44.0.2", "10.42.2.0/24")
            .unwrap();
        cm.set_k0s_phase(
            K0sPhase::Ready,
            Some("head-node".into()),
            Vec::new(),
            Vec::new(),
            false,
        )
        .unwrap();
        wait_for("k0s state applied", || {
            cm.k0s_state().phase == K0sPhase::Ready
                && cm.get_node("n1").and_then(|n| n.k0s_role).is_some()
        });

        // snapshot -> restore into a fresh cluster (log-compaction / HA follower path)
        let data = cm.snapshot_state().unwrap();
        let dir2 = TempDir::new().unwrap();
        let cm2 = test_cluster(&dir2).await;
        cm2.restore_from_snapshot(&data).unwrap();

        // Cluster-wide k0s state must be restored (it is runtime-authoritative).
        let st = cm2.k0s_state();
        assert_eq!(st.phase, K0sPhase::Ready);
        assert_eq!(st.control_plane_node.as_deref(), Some("head-node"));
        // Per-node k0s fields ride snap.nodes.
        let n1 = cm2.get_node("n1").unwrap();
        assert_eq!(n1.k0s_role, Some(K0sRole::Worker));
        assert_eq!(n1.k0s_mesh_ip.as_deref(), Some("10.44.0.2"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn clear_node_k0s_returns_node_to_scheduling() {
        use spur_core::k0s::K0sRole;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);

        cm.assign_node_k0s("n1", K0sRole::Worker, "10.44.0.2", "10.42.2.0/24")
            .unwrap();
        wait_for("role assigned", || {
            cm.get_node("n1").is_some_and(|n| n.is_k0s_reserved())
        });

        cm.clear_node_k0s("n1").unwrap();
        wait_for("role cleared", || {
            cm.get_node("n1").is_some_and(|n| !n.is_k0s_reserved())
        });
        let n1 = cm.get_node("n1").unwrap();
        assert!(n1.k0s_mesh_ip.is_none());
        assert!(n1.k0s_pod_cidr.is_none());

        // A cleared node must place jobs again (teardown reverses the gate).
        let job = submit_and_wait(&cm, basic_spec("post-teardown"));
        let nodes = vec![cm.get_node("n1").unwrap()];
        let state = spur_sched::traits::ClusterState {
            busy_until: &std::collections::HashMap::new(),
            nodes: &nodes,
            partitions: &[],
            reservations: &[],
            topology: None,
        };
        let snap = cm.get_job(job).unwrap();
        cm.update_pending_reasons(&[&snap], &state);
        assert_ne!(
            cm.get_job(job).unwrap().pending_reason,
            PendingReason::K8sReserved
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provision_reassigns_a_readded_node() {
        // The Ready-phase self-heal: a spurd restart deregisters on SIGTERM (the node is REMOVED),
        // then re-registers as a fresh node with no k0s role. Re-running provisioning (which the
        // reconcile loop now does in Ready, not only Provisioning) must re-assign the un-roled node
        // its role + mesh IP + pod CIDR so it rejoins the mesh — without disturbing the others.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "node-a", 4, 8000);
        register_node(&cm, "node-b", 4, 8000);
        wait_for("both registered", || {
            cm.get_node("node-a").is_some() && cm.get_node("node-b").is_some()
        });
        let net = crate::cluster_k8s::ClusterNetworking {
            wg_enabled: true,
            mesh_cidr: "10.44.0.0/16".into(),
            mesh_interface: "spur0".into(),
            pod_cidr: "10.42.0.0/16".into(),
            service_cidr: "10.43.0.0/16".into(),
            cni_mtu: 1450,
            cni: "kuberouter".into(),
            control_plane_node: None,
            provisioning_timeout: std::time::Duration::from_secs(600),
        };
        crate::cluster_k8s::provision_assignments(&cm, &net, &cm.k0s_state()).unwrap();
        wait_for("both assigned", || {
            cm.get_node("node-a").and_then(|n| n.k0s_role).is_some()
                && cm.get_node("node-b").and_then(|n| n.k0s_role).is_some()
        });
        let b_role = cm.get_node("node-b").unwrap().k0s_role;
        let b_ip = cm.get_node("node-b").unwrap().k0s_mesh_ip.clone();
        let b_cidr = cm.get_node("node-b").unwrap().k0s_pod_cidr.clone();
        assert!(b_ip.is_some() && b_cidr.is_some());

        // spurd restart: deregister (remove) then re-register as a fresh, un-roled node.
        cm.remove_node("node-b", true, Some("test restart".into()))
            .unwrap();
        wait_for("node-b removed", || cm.get_node("node-b").is_none());
        register_node(&cm, "node-b", 4, 8000);
        wait_for("node-b re-registered without a role", || {
            cm.get_node("node-b")
                .map(|n| n.k0s_role.is_none())
                .unwrap_or(false)
        });

        // The Ready-phase reconcile re-runs provisioning and heals the un-roled node.
        crate::cluster_k8s::provision_assignments(&cm, &net, &cm.k0s_state()).unwrap();
        wait_for("node-b re-assigned", || {
            cm.get_node("node-b").and_then(|n| n.k0s_role).is_some()
        });
        let b = cm.get_node("node-b").unwrap();
        assert_eq!(b.k0s_role, b_role, "same role after re-add");
        assert_eq!(b.k0s_mesh_ip, b_ip, "same mesh IP reclaimed after re-add");
        assert_eq!(b.k0s_pod_cidr, b_cidr, "same pod CIDR after re-add");
        // The untouched node keeps its assignment (provisioning is idempotent).
        assert!(cm.get_node("node-a").and_then(|n| n.k0s_role).is_some());
    }

    // ---- k0s adopts each node's real WireGuard address as its mesh IP ----

    fn provision_net_calico() -> crate::cluster_k8s::ClusterNetworking {
        crate::cluster_k8s::ClusterNetworking {
            wg_enabled: true,
            mesh_cidr: "10.44.0.0/16".into(),
            mesh_interface: "spur0".into(),
            pod_cidr: "10.42.0.0/16".into(),
            service_cidr: "10.43.0.0/16".into(),
            cni_mtu: 1450,
            cni: "calico".into(),
            control_plane_node: None,
            provisioning_timeout: std::time::Duration::from_secs(600),
        }
    }

    /// Record the HA control-plane set (what `cluster_up` persists) so `provision_assignments`
    /// treats `cp` as controllers and the first as bootstrap.
    fn record_cp_set(cm: &ClusterManager, bootstrap: &str, cp: &[&str]) {
        use spur_core::k0s::K0sPhase;
        cm.set_k0s_phase(
            K0sPhase::Provisioning,
            Some(bootstrap.into()),
            cp.iter().map(|s| s.to_string()).collect(),
            Vec::new(),
            false,
        )
        .unwrap();
        wait_for("cp set recorded", || {
            cm.k0s_state().control_plane_nodes.len() == cp.len()
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provision_adopts_real_address_adversarial_cp_selection() {
        // The core regression: nodes joined the mesh in address order matching hostname
        // order (005=.1 .. 009=.5), but the chosen control-plane set is NOT an alphabetical prefix
        // of that order ({006,007,008}). Each node's k0s_mesh_ip must equal its REAL address — not
        // a pool-allocated value that drifts one-off from reality and corrupts the mesh.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let reals = [
            ("n005", "10.44.0.1"),
            ("n006", "10.44.0.2"),
            ("n007", "10.44.0.3"),
            ("n008", "10.44.0.4"),
            ("n009", "10.44.0.5"),
        ];
        for (name, addr) in reals {
            register_meshed_node(&cm, name, addr);
        }
        // Adversarial: CP set is the middle three, bootstrap 006 (whose real addr is .2, not .1).
        record_cp_set(&cm, "n006", &["n006", "n007", "n008"]);

        crate::cluster_k8s::provision_assignments(&cm, &provision_net_calico(), &cm.k0s_state())
            .unwrap();
        wait_for("all assigned", || {
            reals
                .iter()
                .all(|(n, _)| cm.get_node(n).and_then(|x| x.k0s_role).is_some())
        });

        for (name, addr) in reals {
            let n = cm.get_node(name).unwrap();
            assert_eq!(
                n.k0s_mesh_ip.as_deref(),
                Some(addr),
                "{name}: k0s_mesh_ip must equal its real spur0 address"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provision_adopts_real_address_scrambled_join_order() {
        // Even an alphabetical-prefix CP set mismatches under a pool allocator if physical join
        // order was scrambled (real addresses not in hostname order). Adopting the real address
        // must track physical reality regardless of the name sort.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let reals = [
            ("n005", "10.44.0.5"),
            ("n006", "10.44.0.3"),
            ("n007", "10.44.0.1"),
            ("n008", "10.44.0.4"),
            ("n009", "10.44.0.2"),
        ];
        for (name, addr) in reals {
            register_meshed_node(&cm, name, addr);
        }
        record_cp_set(&cm, "n005", &["n005", "n006", "n007"]);

        crate::cluster_k8s::provision_assignments(&cm, &provision_net_calico(), &cm.k0s_state())
            .unwrap();
        wait_for("all assigned", || {
            reals
                .iter()
                .all(|(n, _)| cm.get_node(n).and_then(|x| x.k0s_role).is_some())
        });

        for (name, addr) in reals {
            let n = cm.get_node(name).unwrap();
            assert_eq!(
                n.k0s_mesh_ip.as_deref(),
                Some(addr),
                "{name}: k0s_mesh_ip must track the real address, not the name sort"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provision_pools_node_with_underlay_comm_address() {
        // A node whose advertised comm address is OUTSIDE mesh_cidr (e.g. a control plane advertising
        // its routable underlay address, as real deployments do) is not reporting a mesh address — it
        // must fall through to pool allocation, NOT be refused. Only an in-mesh address is adopted.
        // (wg_cidr-vs-mesh_cidr config validation is a separate concern.)
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_meshed_node(&cm, "n005", "198.51.100.5"); // underlay, outside 10.44.0.0/16
        register_meshed_node(&cm, "n006", "10.44.0.3"); // in-mesh -> adopted
        record_cp_set(&cm, "n005", &["n005"]);

        crate::cluster_k8s::provision_assignments(&cm, &provision_net_calico(), &cm.k0s_state())
            .unwrap();
        wait_for("both assigned", || {
            cm.get_node("n005").and_then(|n| n.k0s_role).is_some()
                && cm.get_node("n006").and_then(|n| n.k0s_role).is_some()
        });
        // Bootstrap n005 advertised underlay -> pool gives it .1; n006 adopts its real .3.
        assert_eq!(
            cm.get_node("n005").unwrap().k0s_mesh_ip.as_deref(),
            Some("10.44.0.1"),
            "underlay-advertising bootstrap falls back to pool .1"
        );
        assert_eq!(
            cm.get_node("n006").unwrap().k0s_mesh_ip.as_deref(),
            Some("10.44.0.3"),
            "in-mesh node adopts its real address"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provision_refuses_conflicting_real_addresses() {
        // Two unassigned meshed nodes advertising the same real address is a genuine operator mistake:
        // assigning either would push an exclusive AllowedIPs entry for an address two peers claim. Both
        // stay unprovisioned and both carry the reason, but the call still succeeds so any other node in
        // the same tick provisions normally.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_meshed_node(&cm, "n005", "10.44.0.7");
        register_meshed_node(&cm, "n006", "10.44.0.7"); // duplicate
        register_meshed_node(&cm, "n007", "10.44.0.9"); // healthy bystander
        record_cp_set(&cm, "n005", &["n005"]);

        crate::cluster_k8s::provision_assignments(&cm, &provision_net_calico(), &cm.k0s_state())
            .unwrap();

        // The healthy node adopts its real address and provisions despite the conflicting pair.
        wait_for("healthy bystander provisions", || {
            cm.get_node("n007").and_then(|n| n.k0s_role).is_some()
        });
        assert_eq!(
            cm.get_node("n007").unwrap().k0s_mesh_ip.as_deref(),
            Some("10.44.0.9")
        );

        // Neither conflicting node is assigned, and both name each other in an exact-shape reason so
        // `spur k8s status` points at the whole conflict.
        assert!(cm.get_node("n005").and_then(|n| n.k0s_role).is_none());
        assert!(cm.get_node("n006").and_then(|n| n.k0s_role).is_none());
        let want = "network mismatch: mesh address 10.44.0.7 claimed by n005, n006";
        wait_for("both conflicting nodes carry the exact reason", || {
            cm.get_node("n005")
                .and_then(|n| n.k0s_last_error)
                .as_deref()
                == Some(want)
                && cm
                    .get_node("n006")
                    .and_then(|n| n.k0s_last_error)
                    .as_deref()
                    == Some(want)
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provision_refuses_newcomer_colliding_with_assigned_node() {
        // An already-assigned member owns its mesh IP. A newcomer advertising that same address is the
        // one refused — the incumbent keeps its assignment, the newcomer stays out with a reason.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_meshed_node(&cm, "n005", "10.44.0.7");
        record_cp_set(&cm, "n005", &["n005"]);
        crate::cluster_k8s::provision_assignments(&cm, &provision_net_calico(), &cm.k0s_state())
            .unwrap();
        wait_for("incumbent assigned", || {
            cm.get_node("n005").and_then(|n| n.k0s_role).is_some()
        });

        register_meshed_node(&cm, "n006", "10.44.0.7"); // collides with the incumbent's owned address
        crate::cluster_k8s::provision_assignments(&cm, &provision_net_calico(), &cm.k0s_state())
            .unwrap();

        // Incumbent untouched; newcomer refused and told who owns the address.
        assert_eq!(
            cm.get_node("n005").unwrap().k0s_mesh_ip.as_deref(),
            Some("10.44.0.7")
        );
        assert!(cm.get_node("n006").and_then(|n| n.k0s_role).is_none());
        let want = "network mismatch: mesh address 10.44.0.7 already owned by n005";
        wait_for("newcomer carries the ownership reason", || {
            cm.get_node("n006")
                .and_then(|n| n.k0s_last_error)
                .as_deref()
                == Some(want)
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provision_falls_back_to_pool_for_unmeshed_node() {
        // A node not yet on the mesh (no wg pubkey / no real mesh address) keeps pool allocation so
        // the pre-mesh provisioning path still works. Bootstrap still gets .1 in that path.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "node-a", 4, 8000); // 127.0.0.1, no pubkey -> unmeshed
        register_node(&cm, "node-b", 4, 8000);
        record_cp_set(&cm, "node-a", &["node-a"]);

        crate::cluster_k8s::provision_assignments(&cm, &provision_net_calico(), &cm.k0s_state())
            .unwrap();
        wait_for("assigned", || {
            cm.get_node("node-a").and_then(|n| n.k0s_role).is_some()
                && cm.get_node("node-b").and_then(|n| n.k0s_role).is_some()
        });
        // Bootstrap keeps .1 via the pool path; the other gets a distinct pooled address.
        assert_eq!(
            cm.get_node("node-a").unwrap().k0s_mesh_ip.as_deref(),
            Some("10.44.0.1")
        );
        let b = cm.get_node("node-b").unwrap().k0s_mesh_ip.clone().unwrap();
        assert_ne!(b, "10.44.0.1", "second node gets a distinct pooled address");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ready_phase_reconcile_assigns_unroled_node() {
        // The loop wiring: reconcile_phase must run provisioning in the Ready phase, not only
        // Provisioning. An un-roled node present while Ready (e.g. re-added after a spurd restart)
        // must be assigned by a Ready-phase reconcile tick. If Ready were a no-op it would stay
        // un-roled forever (out of the mesh) — the bug this fixes.
        use spur_core::k0s::K0sPhase;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "node-a", 4, 8000);
        wait_for("registered", || cm.get_node("node-a").is_some());
        // Cluster is Ready with the control plane already recorded, but node-a has no k0s role.
        cm.set_k0s_phase(
            K0sPhase::Ready,
            Some("node-a".into()),
            Vec::new(),
            Vec::new(),
            false,
        )
        .unwrap();
        wait_for("phase ready", || cm.k0s_state().phase == K0sPhase::Ready);
        assert!(cm.get_node("node-a").and_then(|n| n.k0s_role).is_none());

        let net = crate::cluster_k8s::ClusterNetworking {
            wg_enabled: true,
            mesh_cidr: "10.44.0.0/16".into(),
            mesh_interface: "spur0".into(),
            pod_cidr: "10.42.0.0/16".into(),
            service_cidr: "10.43.0.0/16".into(),
            cni_mtu: 1450,
            cni: "kuberouter".into(),
            control_plane_node: Some("node-a".into()),
            provisioning_timeout: std::time::Duration::from_secs(600),
        };
        let mut tokens = std::collections::HashMap::new();
        crate::cluster_k8s::reconcile_phase(&cm, &net, &cm.k0s_state(), &mut tokens, false).await;

        wait_for("un-roled node assigned by a Ready-phase tick", || {
            cm.get_node("node-a").and_then(|n| n.k0s_role).is_some()
        });
        assert!(cm.get_node("node-a").and_then(|n| n.k0s_mesh_ip).is_some());
    }

    fn k0s_test_net() -> crate::cluster_k8s::ClusterNetworking {
        crate::cluster_k8s::ClusterNetworking {
            wg_enabled: true,
            mesh_cidr: "10.44.0.0/16".into(),
            mesh_interface: "spur0".into(),
            pod_cidr: "10.42.0.0/16".into(),
            service_cidr: "10.43.0.0/16".into(),
            cni_mtu: 1450,
            cni: "kuberouter".into(),
            control_plane_node: None,
            provisioning_timeout: std::time::Duration::from_secs(600),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provisioning_timeout_degrades_records_reason_keeps_roles() {
        use spur_core::k0s::K0sPhase;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "node-a", 4, 8000);
        wait_for("registered", || cm.get_node("node-a").is_some());
        cm.set_k0s_phase(
            K0sPhase::Provisioning,
            Some("node-a".into()),
            Vec::new(),
            Vec::new(),
            false,
        )
        .unwrap();
        wait_for("phase provisioning", || {
            cm.k0s_state().phase == K0sPhase::Provisioning
        });

        let net = k0s_test_net();
        let mut tokens = std::collections::HashMap::new();
        crate::cluster_k8s::reconcile_phase(&cm, &net, &cm.k0s_state(), &mut tokens, true).await;

        wait_for("degraded", || cm.k0s_state().phase == K0sPhase::Degraded);
        let node = cm.get_node("node-a").unwrap();
        // The node kept its role (recoverable retry) and carries a surfaced reason.
        assert!(node.k0s_role.is_some());
        assert!(node.k0s_last_error.unwrap().contains("not active"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provisioning_within_deadline_stays_provisioning() {
        use spur_core::k0s::K0sPhase;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "node-a", 4, 8000);
        wait_for("registered", || cm.get_node("node-a").is_some());
        cm.set_k0s_phase(
            K0sPhase::Provisioning,
            Some("node-a".into()),
            Vec::new(),
            Vec::new(),
            false,
        )
        .unwrap();
        wait_for("phase provisioning", || {
            cm.k0s_state().phase == K0sPhase::Provisioning
        });

        let net = k0s_test_net();
        let mut tokens = std::collections::HashMap::new();
        crate::cluster_k8s::reconcile_phase(&cm, &net, &cm.k0s_state(), &mut tokens, false).await;

        assert_eq!(cm.k0s_state().phase, K0sPhase::Provisioning);
        assert!(cm.get_node("node-a").unwrap().k0s_last_error.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn set_node_k0s_error_records_and_clears() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "node-a", 4, 8000);
        wait_for("registered", || cm.get_node("node-a").is_some());

        cm.set_node_k0s_error("node-a", Some("boom".into()))
            .unwrap();
        wait_for("error recorded", || {
            cm.get_node("node-a").and_then(|n| n.k0s_last_error) == Some("boom".into())
        });

        cm.set_node_k0s_error("node-a", None).unwrap();
        wait_for("error cleared", || {
            cm.get_node("node-a").unwrap().k0s_last_error.is_none()
        });

        assert!(cm.set_node_k0s_error("ghost", Some("x".into())).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconcile_phase_reports_no_error_for_down_and_degraded() {
        use spur_core::k0s::K0sPhase;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let net = crate::cluster_k8s::ClusterNetworking {
            wg_enabled: true,
            mesh_cidr: "10.44.0.0/16".into(),
            mesh_interface: "spur0".into(),
            pod_cidr: "10.42.0.0/16".into(),
            service_cidr: "10.43.0.0/16".into(),
            cni_mtu: 1450,
            cni: "kuberouter".into(),
            control_plane_node: None,
            provisioning_timeout: std::time::Duration::from_secs(600),
        };
        let mut tokens = std::collections::HashMap::new();
        // A token cached from the torn-down incarnation must not survive the Down tick: reusing it
        // on the next `up` would hand a worker a stale-CA token, failing its join.
        tokens.insert("worker-1".to_string(), "stale-ca-token".to_string());

        let down = spur_core::k0s::K0sClusterState {
            phase: K0sPhase::Down,
            ..Default::default()
        };
        assert!(!crate::cluster_k8s::reconcile_phase(&cm, &net, &down, &mut tokens, false).await);
        assert!(
            tokens.is_empty(),
            "Down reconcile must clear cached join tokens so a rebuild mints fresh ones"
        );

        let degraded = spur_core::k0s::K0sClusterState {
            phase: K0sPhase::Degraded,
            ..Default::default()
        };
        assert!(
            !crate::cluster_k8s::reconcile_phase(&cm, &net, &degraded, &mut tokens, false).await
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn clearing_node_k0s_evicts_its_metric_series() {
        use spur_core::k0s::K0sRole;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "node-a", 4, 8000);
        wait_for("registered", || cm.get_node("node-a").is_some());
        cm.assign_node_k0s("node-a", K0sRole::Worker, "10.44.0.2", "10.42.2.0/24")
            .unwrap();
        // Feed node metrics as a heartbeat would.
        cm.k8s_metrics().set_node_up("test", "node-a", true);
        cm.k8s_metrics().set_node_restart_total("test", "node-a", 1);
        let body = spur_metrics::encode_k8s_metrics(&cm.k8s_cluster_metrics(), &cm.k8s_metrics());
        assert!(body.contains("node=\"node-a\""));

        cm.clear_node_k0s("node-a").unwrap();
        let body = spur_metrics::encode_k8s_metrics(&cm.k8s_cluster_metrics(), &cm.k8s_metrics());
        assert!(
            !body.contains("node=\"node-a\""),
            "cleared node's series must be evicted"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn k8s_cluster_metrics_role_counts_track_assign_clear_and_remove() {
        use spur_core::k0s::K0sRole;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "node-a", 4, 8000);
        register_node(&cm, "node-b", 4, 8000);
        wait_for("both registered", || {
            cm.get_node("node-a").is_some() && cm.get_node("node-b").is_some()
        });

        let counts = |cm: &ClusterManager| {
            let snap = cm.k8s_cluster_metrics();
            (snap.nodes_total, snap.nodes_by_role)
        };
        assert_eq!(counts(&cm).0, 0, "no roles assigned yet");

        cm.assign_node_k0s("node-a", K0sRole::Controller, "10.44.0.2", "10.42.2.0/24")
            .unwrap();
        cm.assign_node_k0s("node-b", K0sRole::Worker, "10.44.0.3", "10.42.3.0/24")
            .unwrap();
        let (total, by_role) = counts(&cm);
        assert_eq!(total, 2);
        assert_eq!(by_role[0], (K0sRole::Controller, 1));
        assert_eq!(by_role[1], (K0sRole::Worker, 1));

        // Re-assigning the same role to the same node must not double-count.
        cm.assign_node_k0s("node-a", K0sRole::Controller, "10.44.0.2", "10.42.2.0/24")
            .unwrap();
        assert_eq!(
            counts(&cm).0,
            2,
            "idempotent re-assign must not inflate the total"
        );

        cm.clear_node_k0s("node-a").unwrap();
        let (total, by_role) = counts(&cm);
        assert_eq!(total, 1);
        assert_eq!(by_role[0], (K0sRole::Controller, 0));
        assert_eq!(by_role[1], (K0sRole::Worker, 1));

        // A force-removed node that still holds a role must not leak into the count.
        cm.remove_node("node-b", true, None).unwrap();
        assert_eq!(
            counts(&cm).0,
            0,
            "removing an assigned node must release its role count"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn set_k0s_phase_counts_attempts_and_transitions() {
        use spur_core::k0s::K0sPhase;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.set_k0s_phase(
            K0sPhase::Provisioning,
            Some("n1".into()),
            Vec::new(),
            Vec::new(),
            false,
        )
        .unwrap();
        wait_for("provisioning", || {
            cm.k0s_state().phase == K0sPhase::Provisioning
        });
        cm.set_k0s_phase(
            K0sPhase::Ready,
            Some("n1".into()),
            Vec::new(),
            Vec::new(),
            false,
        )
        .unwrap();
        wait_for("ready", || cm.k0s_state().phase == K0sPhase::Ready);
        // Re-provision: a second attempt.
        cm.set_k0s_phase(
            K0sPhase::Provisioning,
            Some("n1".into()),
            Vec::new(),
            Vec::new(),
            false,
        )
        .unwrap();
        wait_for("provisioning again", || {
            cm.k0s_state().phase == K0sPhase::Provisioning
        });
        // A no-op set to the same phase must not double-count.
        cm.set_k0s_phase(
            K0sPhase::Provisioning,
            Some("n1".into()),
            Vec::new(),
            Vec::new(),
            false,
        )
        .unwrap();

        let body = spur_metrics::encode_k8s_metrics(&cm.k8s_cluster_metrics(), &cm.k8s_metrics());
        assert!(body.contains(
            "spur_k8s_provision_attempts_total{distribution=\"k0s\",cluster=\"test\"} 2\n"
        ));
        assert!(body.contains("from=\"down\",to=\"provisioning\"} 1\n"));
        assert!(body.contains("from=\"provisioning\",to=\"ready\"} 1\n"));
        assert!(body.contains("from=\"ready\",to=\"provisioning\"} 1\n"));
        // Nothing set Degraded, so no provisioning failure was recorded.
        assert!(body.contains(
            "spur_k8s_provision_failures_total{distribution=\"k0s\",cluster=\"test\"} 0\n"
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provision_assigns_controller_and_worker_for_two_nodes() {
        use spur_core::k0s::K0sRole;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "node-a", 4, 8000);
        register_node(&cm, "node-b", 4, 8000);
        wait_for("both nodes registered", || {
            cm.get_node("node-a").is_some() && cm.get_node("node-b").is_some()
        });

        let net = crate::cluster_k8s::ClusterNetworking {
            wg_enabled: true,
            mesh_cidr: "10.44.0.0/16".into(),
            mesh_interface: "spur0".into(),
            pod_cidr: "10.42.0.0/16".into(),
            service_cidr: "10.43.0.0/16".into(),
            cni_mtu: 1450,
            cni: "kuberouter".into(),
            control_plane_node: None,
            provisioning_timeout: std::time::Duration::from_secs(600),
        };
        crate::cluster_k8s::provision_assignments(&cm, &net, &cm.k0s_state()).unwrap();
        wait_for("both nodes assigned k0s roles", || {
            cm.get_node("node-a").and_then(|n| n.k0s_role).is_some()
                && cm.get_node("node-b").and_then(|n| n.k0s_role).is_some()
        });

        // Two nodes -> the deterministic control-plane (lexically-first, node-a) is a Controller
        // (NOT Single, which would never exercise the worker token-mint/join path); node-b is a
        // Worker. Each gets a distinct mesh IP + pod /24, and the control-plane choice is recorded.
        let a = cm.get_node("node-a").unwrap();
        let b = cm.get_node("node-b").unwrap();
        assert_eq!(a.k0s_role, Some(K0sRole::Controller));
        assert_eq!(b.k0s_role, Some(K0sRole::Worker));
        assert!(a.k0s_mesh_ip.is_some() && b.k0s_mesh_ip.is_some());
        assert_ne!(a.k0s_mesh_ip, b.k0s_mesh_ip);
        assert_ne!(a.k0s_pod_cidr, b.k0s_pod_cidr);
        assert_eq!(cm.k0s_state().control_plane_node.as_deref(), Some("node-a"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provision_only_roles_scoped_member_nodes() {
        use spur_core::k0s::K0sPhase;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        for name in ["node-a", "node-b", "node-c"] {
            register_node(&cm, name, 4, 8000);
        }
        wait_for("all registered", || {
            ["node-a", "node-b", "node-c"]
                .iter()
                .all(|n| cm.get_node(n).is_some())
        });
        // Scope to a/b only; node-c is out of scope and must never get a role (stays schedulable).
        cm.set_k0s_phase(
            K0sPhase::Provisioning,
            Some("node-a".into()),
            vec!["node-a".into()],
            vec!["node-a".into(), "node-b".into()],
            false,
        )
        .unwrap();
        wait_for("scope recorded", || cm.k0s_state().member_nodes.len() == 2);

        let net = crate::cluster_k8s::ClusterNetworking {
            wg_enabled: true,
            mesh_cidr: "10.44.0.0/16".into(),
            mesh_interface: "spur0".into(),
            pod_cidr: "10.42.0.0/16".into(),
            service_cidr: "10.43.0.0/16".into(),
            cni_mtu: 1450,
            cni: "kuberouter".into(),
            control_plane_node: Some("node-a".into()),
            provisioning_timeout: std::time::Duration::from_secs(600),
        };
        crate::cluster_k8s::provision_assignments(&cm, &net, &cm.k0s_state()).unwrap();
        wait_for("scoped nodes assigned", || {
            cm.get_node("node-a").and_then(|n| n.k0s_role).is_some()
                && cm.get_node("node-b").and_then(|n| n.k0s_role).is_some()
        });
        assert!(
            cm.get_node("node-c").and_then(|n| n.k0s_role).is_none(),
            "out-of-scope node must stay un-roled and schedulable"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn k0s_member_nodes_add_and_remove_apply() {
        use spur_core::k0s::K0sPhase;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.set_k0s_phase(
            K0sPhase::Ready,
            Some("cp".into()),
            vec!["cp".into()],
            vec!["cp".into(), "w1".into()],
            false,
        )
        .unwrap();
        wait_for("scope recorded", || cm.k0s_state().member_nodes.len() == 2);

        // Add unions in (sorted, deduped).
        cm.add_k0s_member_nodes(vec!["w2".into(), "w1".into()])
            .unwrap();
        wait_for("w2 added", || cm.k0s_state().member_nodes.len() == 3);
        assert_eq!(cm.k0s_state().member_nodes, vec!["cp", "w1", "w2"]);

        // Remove subtracts (idempotent — removing an absent node is a no-op).
        cm.remove_k0s_member_nodes(vec!["w1".into(), "ghost".into()])
            .unwrap();
        wait_for("w1 removed", || cm.k0s_state().member_nodes.len() == 2);
        assert_eq!(cm.k0s_state().member_nodes, vec!["cp", "w2"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn down_clears_member_scope_and_control_plane() {
        use spur_core::k0s::K0sPhase;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.set_k0s_phase(
            K0sPhase::Provisioning,
            Some("node-a".into()),
            vec!["node-a".into()],
            vec!["node-a".into(), "node-b".into()],
            false,
        )
        .unwrap();
        wait_for("scope recorded", || {
            cm.k0s_state().member_nodes.len() == 2 && !cm.k0s_state().controllers().is_empty()
        });
        // A plain down (reset=false) must clear the recorded scope + control plane so the next up
        // starts clean rather than silently reusing stale identity.
        cm.set_k0s_phase(K0sPhase::Down, None, Vec::new(), Vec::new(), false)
            .unwrap();
        wait_for("state cleared on down", || {
            let s = cm.k0s_state();
            s.member_nodes.is_empty()
                && s.control_plane_node.is_none()
                && s.controllers().is_empty()
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn down_clears_even_if_wal_op_carries_stale_fields() {
        use spur_core::k0s::K0sPhase;
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        // No production caller passes non-empty fields alongside Down, but the WAL apply must not
        // depend on that: it should clear unconditionally rather than set-then-clear.
        cm.set_k0s_phase(
            K0sPhase::Down,
            Some("stale-cp".into()),
            vec!["stale-cp".into()],
            vec!["stale-a".into(), "stale-b".into()],
            false,
        )
        .unwrap();
        wait_for("state cleared despite non-empty op fields", || {
            let s = cm.k0s_state();
            s.member_nodes.is_empty()
                && s.control_plane_node.is_none()
                && s.controllers().is_empty()
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provision_assigns_three_controllers_for_ha_set() {
        use spur_core::k0s::{K0sPhase, K0sRole};
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        for name in ["cp-a", "cp-b", "cp-c", "w-d"] {
            register_node(&cm, name, 4, 8000);
        }
        wait_for("all registered", || {
            ["cp-a", "cp-b", "cp-c", "w-d"]
                .iter()
                .all(|n| cm.get_node(n).is_some())
        });
        // A 3-CP HA set recorded up front (as `cluster_up` does), bootstrap cp-a first.
        cm.set_k0s_phase(
            K0sPhase::Provisioning,
            Some("cp-a".into()),
            vec!["cp-a".into(), "cp-b".into(), "cp-c".into()],
            Vec::new(),
            false,
        )
        .unwrap();
        wait_for("cp set recorded", || {
            cm.k0s_state().controllers().len() == 3
        });

        let net = crate::cluster_k8s::ClusterNetworking {
            wg_enabled: true,
            mesh_cidr: "10.44.0.0/16".into(),
            mesh_interface: "spur0".into(),
            pod_cidr: "10.42.0.0/16".into(),
            service_cidr: "10.43.0.0/16".into(),
            cni_mtu: 1450,
            cni: "kuberouter".into(),
            control_plane_node: None,
            provisioning_timeout: std::time::Duration::from_secs(600),
        };
        crate::cluster_k8s::provision_assignments(&cm, &net, &cm.k0s_state()).unwrap();
        wait_for("all four assigned", || {
            ["cp-a", "cp-b", "cp-c", "w-d"]
                .iter()
                .all(|n| cm.get_node(n).and_then(|x| x.k0s_role).is_some())
        });

        // All three CP nodes are Controllers; the fourth is a Worker. The bootstrap keeps `.1`.
        for cp in ["cp-a", "cp-b", "cp-c"] {
            assert_eq!(
                cm.get_node(cp).unwrap().k0s_role,
                Some(K0sRole::Controller),
                "{cp} is a controller"
            );
        }
        assert_eq!(cm.get_node("w-d").unwrap().k0s_role, Some(K0sRole::Worker));
        assert_eq!(
            cm.get_node("cp-a").unwrap().k0s_mesh_ip.as_deref(),
            Some("10.44.0.1"),
            "bootstrap holds .1"
        );
        // Every node has a distinct mesh IP.
        let ips: std::collections::HashSet<_> = ["cp-a", "cp-b", "cp-c", "w-d"]
            .iter()
            .map(|n| cm.get_node(n).unwrap().k0s_mesh_ip.clone())
            .collect();
        assert_eq!(ips.len(), 4, "distinct mesh IPs");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provision_derives_bootstrap_from_set_when_singular_absent() {
        use spur_core::k0s::{K0sPhase, K0sRole};
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        for name in ["cp-a", "cp-b", "cp-c"] {
            register_node(&cm, name, 4, 8000);
        }
        wait_for("all registered", || {
            ["cp-a", "cp-b", "cp-c"]
                .iter()
                .all(|n| cm.get_node(n).is_some())
        });
        // HA set recorded bootstrap-first (cp-b) with the singular field unset. The first-of-set must
        // hold `.1` — cp-b, NOT the sorted-first cp-a, so this fails if bootstrap ignores the set.
        cm.set_k0s_phase(
            K0sPhase::Provisioning,
            None,
            vec!["cp-b".into(), "cp-a".into(), "cp-c".into()],
            Vec::new(),
            false,
        )
        .unwrap();
        wait_for("cp set recorded", || {
            cm.k0s_state().controllers().len() == 3
        });

        let net = crate::cluster_k8s::ClusterNetworking {
            wg_enabled: true,
            mesh_cidr: "10.44.0.0/16".into(),
            mesh_interface: "spur0".into(),
            pod_cidr: "10.42.0.0/16".into(),
            service_cidr: "10.43.0.0/16".into(),
            cni_mtu: 1450,
            cni: "kuberouter".into(),
            control_plane_node: None,
            provisioning_timeout: std::time::Duration::from_secs(600),
        };
        crate::cluster_k8s::provision_assignments(&cm, &net, &cm.k0s_state()).unwrap();
        wait_for("all assigned", || {
            ["cp-a", "cp-b", "cp-c"]
                .iter()
                .all(|n| cm.get_node(n).and_then(|x| x.k0s_role).is_some())
        });

        for cp in ["cp-a", "cp-b", "cp-c"] {
            assert_eq!(cm.get_node(cp).unwrap().k0s_role, Some(K0sRole::Controller));
        }
        assert_eq!(
            cm.get_node("cp-b").unwrap().k0s_mesh_ip.as_deref(),
            Some("10.44.0.1"),
            "bootstrap (first of recorded set) holds .1"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_from_snapshot_drops_stale_live_partition() {
        // A partition present in the target's live memory but absent from the
        // snapshot (and not tombstoned) must not survive a snapshot install —
        // otherwise a follower diverges from the leader's partition table.
        let src = TempDir::new().unwrap();
        let cm = test_cluster(&src).await;
        let data = cm.snapshot_state().unwrap();

        let dst = TempDir::new().unwrap();
        let cm2 = test_cluster(&dst).await;
        cm2.apply_operation(&WalOperation::PartitionCreate {
            partition: gpu_partition(),
        });
        assert!(cm2.get_partitions().iter().any(|p| p.name == "gpu"));

        cm2.restore_from_snapshot(&data).unwrap();
        assert!(
            !cm2.get_partitions().iter().any(|p| p.name == "gpu"),
            "stale live partition must be gone after restore"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_from_snapshot_rebuilds_k0s_role_counts() {
        // The target never applies the source's NodeK0sAssign op itself — it only loads the
        // already-assigned Node from the snapshot's node list — so this only passes if restore
        // recomputes k0s_role_counts from scratch rather than relying on incremental apply().
        use spur_core::k0s::K0sRole;
        let src = TempDir::new().unwrap();
        let cm = test_cluster(&src).await;
        register_node(&cm, "node-a", 4, 8000);
        wait_for("registered", || cm.get_node("node-a").is_some());
        cm.assign_node_k0s("node-a", K0sRole::Controller, "10.44.0.2", "10.42.2.0/24")
            .unwrap();
        let data = cm.snapshot_state().unwrap();

        let dst = TempDir::new().unwrap();
        let cm2 = test_cluster(&dst).await;
        cm2.restore_from_snapshot(&data).unwrap();

        let snap = cm2.k8s_cluster_metrics();
        assert_eq!(snap.nodes_total, 1);
        assert_eq!(snap.nodes_by_role[0], (K0sRole::Controller, 1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_from_pre_partition_snapshot_keeps_config_baseline() {
        // An old snapshot predating partition support has no `partitions` field
        // (serde-default → empty). Restore must fall back to the config
        // baseline, not wipe all partitions.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let baseline: Vec<String> = cm.get_partitions().iter().map(|p| p.name.clone()).collect();
        assert!(!baseline.is_empty(), "test config must define a partition");

        let mut snap: serde_json::Value =
            serde_json::from_slice(&cm.snapshot_state().unwrap()).unwrap();
        snap.as_object_mut().unwrap().remove("partitions");
        let data = serde_json::to_vec(&snap).unwrap();

        cm.restore_from_snapshot(&data).unwrap();
        let after: Vec<String> = cm.get_partitions().iter().map(|p| p.name.clone()).collect();
        assert_eq!(
            after, baseline,
            "config baseline must survive an old snapshot"
        );
    }

    // An authoritative empty set (leader deleted them all) must install verbatim,
    // not reseed from local config — the case `is_empty()` conflated with legacy.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_from_authoritative_empty_snapshot_wipes_partitions() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        assert!(
            !cm.get_partitions().is_empty(),
            "test config must seed a partition"
        );

        // A leader snapshot with an explicit empty (Some(vec![])) partition set.
        let mut snap: serde_json::Value =
            serde_json::from_slice(&cm.snapshot_state().unwrap()).unwrap();
        snap.as_object_mut()
            .unwrap()
            .insert("partitions".into(), serde_json::json!([]));
        let data = serde_json::to_vec(&snap).unwrap();

        cm.restore_from_snapshot(&data).unwrap();
        assert!(
            cm.get_partitions().is_empty(),
            "authoritative empty set must wipe local partitions, not reseed from config"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_from_snapshot_rejects_corrupt_data() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        assert!(cm.restore_from_snapshot(b"not valid json").is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hold_and_release_job() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let id = submit_and_wait(&cm, basic_spec("holdme"));

        cm.hold_job(id).unwrap();
        wait_for("hold applied", || {
            cm.get_job(id).is_some_and(|j| j.priority == 0)
        });
        let job = cm.get_job(id).unwrap();
        assert_eq!(job.priority, 0);
        assert_eq!(job.pending_reason, PendingReason::Held);

        cm.release_job(id).unwrap();
        wait_for("release applied", || {
            cm.get_job(id).is_some_and(|j| j.priority > 0)
        });
        let job = cm.get_job(id).unwrap();
        assert_eq!(job.priority, 1000);
        assert_eq!(job.pending_reason, PendingReason::Priority);
    }

    // hold_job_for_launch_failure is confirm_dispatch_on_nodes's Pending-
    // compatible equivalent of the old Running->Failed->Held detour: same end
    // state as a plain hold_job (Pending, priority 0, PendingReason::Held),
    // plus the launch-failure description, and it must refuse a job that
    // already started (unlike a fresh Pending job, that would silently do the
    // wrong thing by holding it mid-run).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hold_job_for_launch_failure_holds_a_pending_job_with_its_description() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let id = submit_and_wait(&cm, basic_spec("prolog-hold"));

        cm.hold_job_for_launch_failure(id, None).unwrap();
        wait_for("hold applied", || {
            cm.get_job(id).is_some_and(|j| j.priority == 0)
        });

        let job = cm.get_job(id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.priority, 0);
        assert_eq!(job.pending_reason, PendingReason::Held);
        assert_eq!(
            job.state_reason_display(),
            LAUNCH_FAILURE_HELD_DESC,
            "must carry the same description the old post-Running hold used"
        );

        // Releasing it behaves exactly like releasing a plain hold_job.
        cm.release_job(id).unwrap();
        wait_for("release applied", || {
            cm.get_job(id).is_some_and(|j| j.priority > 0)
        });
        let job = cm.get_job(id).unwrap();
        assert_eq!(
            job.pending_reason_desc, None,
            "the release must clear the description with the reason"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hold_job_for_launch_failure_refuses_a_job_that_already_started() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("already-running"));
        cm.start_job(
            id,
            vec!["n1".into()],
            scalar_alloc(1, 1000),
            per_node_for(&["n1"], scalar_alloc(1, 1000)),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);

        assert!(cm.hold_job_for_launch_failure(id, None).is_err());
    }

    // backoff_pending_job_after_dispatch_failure is confirm_dispatch_on_nodes's
    // Pending-safe equivalent of requeue_after_launch_failure's backoff: both
    // are no-ops rather than errors for a job that's already moved on by the
    // time the caller gets around to backing it off (e.g. a concurrent
    // cancel, or a stale/unknown job_id) — the caller only has a job_id and
    // no reason to assume it's still exactly as it left it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backoff_pending_job_after_dispatch_failure_is_a_noop_for_a_missing_job() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        assert!(cm.backoff_pending_job_after_dispatch_failure(999).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backoff_pending_job_after_dispatch_failure_is_a_noop_once_the_job_left_pending() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let id = submit_and_wait(&cm, basic_spec("backoff-already-cancelled"));
        cm.cancel_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Cancelled);

        assert!(cm.backoff_pending_job_after_dispatch_failure(id).is_ok());
        assert_eq!(cm.get_job(id).unwrap().state, JobState::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backoff_pending_job_after_dispatch_failure_applies_backoff_and_bumps_requeue_count() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let id = submit_and_wait(&cm, basic_spec("backoff-applies"));

        cm.backoff_pending_job_after_dispatch_failure(id).unwrap();
        wait_for("backoff applied", || {
            cm.get_job(id).is_some_and(|j| j.requeue_count == 1)
        });

        let job = cm.get_job(id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.pending_reason, PendingReason::JobLaunchFailure);
        assert!(job.spec.begin_time.is_some_and(|t| t > Utc::now()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_dispatch_backoff_preserves_launch_failure_detail() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let id = submit_and_wait(&cm, basic_spec("backoff-detail"));
        cm.set_job_launch_failure_detail(id, "PMIx prepare failed: n2: timeout".into())
            .unwrap();

        cm.backoff_pending_job_after_dispatch_failure(id).unwrap();
        wait_for("backoff applied", || {
            cm.get_job(id).is_some_and(|j| j.requeue_count == 1)
        });

        let job = cm.get_job(id).unwrap();
        assert_eq!(job.pending_reason, PendingReason::JobLaunchFailure);
        assert_eq!(
            job.launch_failure_detail.as_deref(),
            Some("PMIx prepare failed: n2: timeout")
        );
        assert_eq!(
            job.state_reason(),
            "JobLaunchFailure (PMIx prepare failed: n2: timeout)"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn second_dispatch_backoff_preserves_launch_failure_detail() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let id = submit_and_wait(&cm, basic_spec("backoff-detail-twice"));

        cm.set_job_launch_failure_detail(id, "PMIx prepare failed: n1: timeout".into())
            .unwrap();
        cm.backoff_pending_job_after_dispatch_failure(id).unwrap();
        wait_for("first backoff applied", || {
            cm.get_job(id).is_some_and(|j| j.requeue_count == 1)
        });

        cm.set_job_launch_failure_detail(id, "PMIx prepare failed: n2: timeout".into())
            .unwrap();
        cm.backoff_pending_job_after_dispatch_failure(id).unwrap();
        wait_for("second backoff applied", || {
            cm.get_job(id).is_some_and(|j| j.requeue_count == 2)
        });

        let job = cm.get_job(id).unwrap();
        assert_eq!(job.pending_reason, PendingReason::JobLaunchFailure);
        assert_eq!(
            job.launch_failure_detail.as_deref(),
            Some("PMIx prepare failed: n2: timeout")
        );
        assert_eq!(
            job.state_reason(),
            "JobLaunchFailure (PMIx prepare failed: n2: timeout)"
        );
    }

    // The apply-level checks mirror the public method's, guarding against a
    // narrower race: the job matched (existed, was Pending) when the public
    // method read it, but no longer does by the time this specific WAL entry
    // is actually applied (a state change committed in between). Exercised
    // directly against apply_operation, bypassing the public method's own
    // up-front checks, so this is testing the apply arm's defense
    // independently of whether the caller's own check could have caught it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_dispatch_backoff_apply_is_a_noop_for_a_missing_job() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.apply_operation(&WalOperation::JobDispatchBackoff {
            job_id: 999,
            begin_time: Utc::now(),
        });
        assert!(cm.get_job(999).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_dispatch_backoff_apply_is_a_noop_once_the_job_left_pending() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let id = submit_and_wait(&cm, basic_spec("backoff-apply-already-cancelled"));
        cm.cancel_job(id, "testuser").unwrap();
        settle(&cm, id, JobState::Cancelled);
        let requeue_count_before = cm.get_job(id).unwrap().requeue_count;

        cm.apply_operation(&WalOperation::JobDispatchBackoff {
            job_id: id,
            begin_time: Utc::now(),
        });

        let job = cm.get_job(id).unwrap();
        assert_eq!(job.state, JobState::Cancelled);
        assert_eq!(job.requeue_count, requeue_count_before);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_job_priority() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let id = submit_and_wait(&cm, basic_spec("prio"));

        cm.update_job(id, None, Some(5000), None, None, None, None)
            .unwrap();
        wait_for("priority updated", || {
            cm.get_job(id).is_some_and(|j| j.priority == 5000)
        });
        assert_eq!(cm.get_job(id).unwrap().priority, 5000);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_job_qos_validates_against_cache() {
        // `scontrol update job qos=` must not be a second door to the bypass:
        // unknown and empty QOS are rejected, leaving the job's QOS unchanged.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.qos_cache().insert(Qos {
            name: "highprio".into(),
            ..Default::default()
        });
        let id = submit_and_wait(&cm, basic_spec("q"));
        assert_eq!(cm.get_job(id).unwrap().spec.qos, None);

        // Unknown QOS rejected.
        let err = cm
            .update_job(id, None, None, None, None, None, Some("ghost".into()))
            .unwrap_err();
        assert!(err.to_string().contains("QOS 'ghost' does not exist"));
        assert!(!err.to_string().contains("accounting may not be enabled"));
        assert_eq!(cm.get_job(id).unwrap().spec.qos, None);

        // Empty QOS (clear-to-limitless) rejected.
        let err = cm
            .update_job(id, None, None, None, None, None, Some(String::new()))
            .unwrap_err();
        assert!(err.to_string().contains("cannot clear a job's QOS"));
        assert_eq!(cm.get_job(id).unwrap().spec.qos, None);

        // A valid QOS is applied.
        cm.update_job(id, None, None, None, None, None, Some("highprio".into()))
            .unwrap();
        assert_eq!(
            cm.get_job(id).unwrap().spec.qos.as_deref(),
            Some("highprio")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_job_qos_hints_when_cache_unloaded() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let id = submit_and_wait(&cm, basic_spec("q"));
        assert_eq!(cm.get_job(id).unwrap().spec.qos, None);

        let err = cm
            .update_job(id, None, None, None, None, None, Some("any-qos".into()))
            .unwrap_err();
        assert!(err.to_string().contains("QOS 'any-qos' does not exist"));
        assert!(err.to_string().contains("accounting may not be enabled"));
        assert_eq!(cm.get_job(id).unwrap().spec.qos, None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_job_qos_rejects_unauthorized_association_qos() {
        // `scontrol update job qos=` must enforce the same per-association
        // authorization as submission (SPUR-101).
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.qos_cache().insert(Qos {
            name: "highprio".into(),
            ..Default::default()
        });
        cm.qos_cache().insert(Qos {
            name: "other-teams-qos".into(),
            ..Default::default()
        });
        cm.association_cache()
            .insert_default_qos("testuser", "research", "highprio");
        let mut spec = basic_spec("q");
        spec.account = Some("research".into());
        let id = submit_and_wait(&cm, spec);
        // Submission already applied the association's own default QOS.
        assert_eq!(
            cm.get_job(id).unwrap().spec.qos.as_deref(),
            Some("highprio")
        );

        let err = cm
            .update_job(
                id,
                None,
                None,
                None,
                None,
                None,
                Some("other-teams-qos".into()),
            )
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("QOS 'other-teams-qos' is not permitted"));
        assert_eq!(
            cm.get_job(id).unwrap().spec.qos.as_deref(),
            Some("highprio"),
            "rejected update must leave the existing QOS untouched"
        );

        // The association's own default QOS is still accepted explicitly.
        cm.update_job(id, None, None, None, None, None, Some("highprio".into()))
            .unwrap();
        assert_eq!(
            cm.get_job(id).unwrap().spec.qos.as_deref(),
            Some("highprio")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_job_qos_rejects_bypass_via_unaffiliated_account() {
        // Pairing the QOS change with an account the user has no recorded
        // association for must not fall back to "nothing to check" — that
        // was a live bypass of the authorization this PR adds.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.qos_cache().insert(Qos {
            name: "highprio".into(),
            ..Default::default()
        });
        cm.qos_cache().insert(Qos {
            name: "other-teams-qos".into(),
            ..Default::default()
        });
        cm.association_cache()
            .insert_default_qos("testuser", "research", "highprio");
        let id = submit_and_wait(&cm, basic_spec("q"));

        let err = cm
            .update_job(
                id,
                None,
                None,
                None,
                None,
                Some("made-up-account".into()),
                Some("other-teams-qos".into()),
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("not associated with account"),
            "expected an association error, got {err}"
        );
        assert_eq!(cm.get_job(id).unwrap().spec.qos, None);

        // An empty account string must fall back to the job's existing
        // account, not to "no account given" (which would skip
        // authorization entirely).
        let mut spec = basic_spec("q2");
        spec.account = Some("research".into());
        let id2 = submit_and_wait(&cm, spec);
        let err = cm
            .update_job(
                id2,
                None,
                None,
                None,
                None,
                Some(String::new()),
                Some("other-teams-qos".into()),
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("is not permitted"),
            "expected re-enforcement against the job's own account 'research', got {err}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_job_qos_allows_any_member_of_the_allow_list() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.qos_cache().insert(Qos {
            name: "a".into(),
            ..Default::default()
        });
        cm.qos_cache().insert(Qos {
            name: "b".into(),
            ..Default::default()
        });
        cm.qos_cache().insert(Qos {
            name: "other-teams-qos".into(),
            ..Default::default()
        });
        cm.association_cache()
            .insert_allowed_qos("testuser", "research", &["a", "b"]);
        let mut spec = basic_spec("q");
        spec.account = Some("research".into());
        let id = submit_and_wait(&cm, spec);

        cm.update_job(id, None, None, None, None, None, Some("b".into()))
            .unwrap();
        assert_eq!(cm.get_job(id).unwrap().spec.qos.as_deref(), Some("b"));

        let err = cm
            .update_job(
                id,
                None,
                None,
                None,
                None,
                None,
                Some("other-teams-qos".into()),
            )
            .unwrap_err();
        assert!(err.to_string().contains("is not permitted"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_node_state() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);

        cm.update_node_state("n1", NodeState::Drain, Some("maint".into()), Some(1000))
            .unwrap();
        wait_for("node drain applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Drain)
        });
        let node = cm.get_node("n1").unwrap();
        assert_eq!(node.state, NodeState::Drain);
        assert_eq!(node.state_reason, Some("maint".into()));
        assert_eq!(node.reason_uid, Some(1000));
        assert!(
            node.reason_time.is_some(),
            "set-time recorded with the reason"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_node_state_resume_clears_reason_attribution() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);

        cm.update_node_state("n1", NodeState::Drain, Some("maint".into()), Some(1000))
            .unwrap();
        wait_for("drain applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.reason_uid == Some(1000))
        });

        cm.update_node_state("n1", NodeState::Idle, None, None)
            .unwrap();
        wait_for("resume applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Idle)
        });
        let node = cm.get_node("n1").unwrap();
        assert_eq!(node.state_reason, None);
        assert_eq!(node.reason_uid, None, "resume clears the setter uid");
        assert_eq!(node.reason_time, None, "resume clears the set-time");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heartbeat_down_attributes_reason_to_system_uid() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "stale", 4, 8000);
        if let Some(node) = cm.nodes.write().get_mut("stale") {
            node.last_heartbeat = Some(Utc::now() - chrono::Duration::seconds(200));
        }

        cm.check_node_health(90, super::MarkDownPolicy::Allowed);
        wait_for("health down applied", || {
            cm.get_node("stale")
                .is_some_and(|n| n.state == NodeState::Down)
        });
        let node = cm.get_node("stale").unwrap();
        assert_eq!(node.state_reason.as_deref(), Some("Not responding"));
        assert_eq!(node.reason_uid, Some(0), "system reason is uid 0");
        assert!(node.reason_time.is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heartbeat_down_preserves_admin_hold_attribution() {
        // A drained node (admin hold) that then stops heartbeating keeps the
        // operator's reason and its original uid/time, not the system uid.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);

        cm.update_node_state("n1", NodeState::Drain, Some("hw swap".into()), Some(1000))
            .unwrap();
        wait_for("drain applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.reason_uid == Some(1000))
        });

        cm.apply_health_actions(vec![super::HealthAction::MarkDown {
            name: "n1".into(),
            old_state: NodeState::Drain,
            admin_locked: true,
        }]);
        wait_for("down applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Down)
        });
        let node = cm.get_node("n1").unwrap();
        assert_eq!(node.state_reason.as_deref(), Some("hw swap"));
        assert_eq!(node.reason_uid, Some(1000), "admin-hold uid preserved");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn check_node_health_marks_stale_down() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "stale", 4, 8000);

        // Set last_heartbeat far in the past
        if let Some(node) = cm.nodes.write().get_mut("stale") {
            node.last_heartbeat = Some(Utc::now() - chrono::Duration::seconds(200));
        }

        cm.check_node_health(90, super::MarkDownPolicy::Allowed);
        wait_for("health check applied", || {
            cm.get_node("stale")
                .is_some_and(|n| n.state == NodeState::Down)
        });
        let node = cm.get_node("stale").unwrap();
        assert_eq!(node.state, NodeState::Down);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn check_node_health_recovers_allocated_node_as_allocated() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "recovering", 4, 8000);

        let id = submit_and_wait(&cm, basic_spec("running-job"));
        let alloc = scalar_alloc(4, 8000);
        cm.start_job(
            id,
            vec!["recovering".into()],
            alloc.clone(),
            per_node_for(&["recovering"], alloc),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);

        if let Some(node) = cm.nodes.write().get_mut("recovering") {
            node.state = NodeState::Down;
            node.last_heartbeat = Some(Utc::now());
        }

        cm.check_node_health(90, super::MarkDownPolicy::Allowed);
        wait_for("node recovery applied", || {
            cm.get_node("recovering")
                .is_some_and(|node| node.state == NodeState::Allocated)
        });
        assert_eq!(
            cm.get_node("recovering").unwrap().state,
            NodeState::Allocated
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn suppressed_mark_down_spares_a_stale_node_then_allowed_marks_it() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "stale", 4, 8000);
        if let Some(node) = cm.nodes.write().get_mut("stale") {
            node.last_heartbeat = Some(Utc::now() - chrono::Duration::seconds(200));
        }

        cm.check_node_health(90, super::MarkDownPolicy::Suppressed);
        assert_eq!(
            cm.get_node("stale").unwrap().state,
            NodeState::Idle,
            "stale node must survive the leadership grace window"
        );

        cm.check_node_health(90, super::MarkDownPolicy::Allowed);
        wait_for("down applied after grace", || {
            cm.get_node("stale")
                .is_some_and(|n| n.state == NodeState::Down)
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn suppressed_mark_down_still_recovers_a_node_whose_heartbeat_returned() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "back", 4, 8000);
        cm.apply_health_actions(vec![super::HealthAction::MarkDown {
            name: "back".into(),
            old_state: NodeState::Idle,
            admin_locked: false,
        }]);
        wait_for("down applied", || {
            cm.get_node("back")
                .is_some_and(|n| n.state == NodeState::Down)
        });
        if let Some(node) = cm.nodes.write().get_mut("back") {
            node.last_heartbeat = Some(Utc::now());
        }

        cm.check_node_health(90, super::MarkDownPolicy::Suppressed);
        wait_for("recovered during grace", || {
            cm.get_node("back")
                .is_some_and(|n| n.state == NodeState::Idle)
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn suppressed_mark_down_does_not_evict_running_jobs() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("keep-running"));
        let alloc = scalar_alloc(4, 8000);
        cm.start_job(
            id,
            vec!["n1".into()],
            alloc.clone(),
            per_node_for(&["n1"], alloc),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);

        if let Some(node) = cm.nodes.write().get_mut("n1") {
            node.last_heartbeat = Some(Utc::now() - chrono::Duration::seconds(200));
        }

        let evicted = cm.check_node_health(90, super::MarkDownPolicy::Suppressed);
        assert!(evicted.is_empty(), "grace window must not evict jobs");
        assert_eq!(cm.get_job(id).unwrap().state, JobState::Running);
        assert_eq!(
            cm.get_job(id).unwrap().allocated_nodes,
            vec!["n1".to_string()]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_preserves_an_admin_drain_raced_against_the_health_check() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);

        cm.apply_health_actions(vec![super::HealthAction::MarkDown {
            name: "n1".into(),
            old_state: NodeState::Idle,
            admin_locked: false,
        }]);
        wait_for("down applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Down)
        });

        cm.update_node_state("n1", NodeState::Drain, Some("hw swap".into()), None)
            .unwrap();
        wait_for("drain applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Drain)
        });

        cm.apply_health_actions(vec![super::HealthAction::Recover {
            name: "n1".into(),
            old_state: NodeState::Down,
        }]);

        let node = cm.get_node("n1").unwrap();
        assert_eq!(
            node.state,
            NodeState::Drain,
            "recovery must not un-drain an administratively drained node"
        );
        assert!(node.admin_locked);
        assert_eq!(node.state_reason, Some("hw swap".into()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_clears_the_liveness_reason_on_an_unlocked_node() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);

        cm.apply_health_actions(vec![super::HealthAction::MarkDown {
            name: "n1".into(),
            old_state: NodeState::Idle,
            admin_locked: false,
        }]);
        wait_for("down applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Down)
        });

        cm.apply_health_actions(vec![super::HealthAction::Recover {
            name: "n1".into(),
            old_state: NodeState::Down,
        }]);
        wait_for("recovery applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Idle)
        });
        let node = cm.get_node("n1").unwrap();
        assert!(!node.admin_locked);
        assert_eq!(node.state_reason, None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mark_down_preserves_an_admin_drain_reason_when_heartbeat_also_times_out() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("draining"));
        let alloc = scalar_alloc(4, 8000);
        cm.start_job(
            id,
            vec!["n1".into()],
            alloc.clone(),
            per_node_for(&["n1"], alloc),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);

        cm.update_node_state("n1", NodeState::Drain, Some("hw swap".into()), None)
            .unwrap();
        wait_for("draining applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Draining)
        });

        cm.apply_health_actions(vec![super::HealthAction::MarkDown {
            name: "n1".into(),
            old_state: NodeState::Draining,
            admin_locked: true,
        }]);
        wait_for("down applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Down)
        });

        let node = cm.get_node("n1").unwrap();
        assert_eq!(
            node.state_reason,
            Some("hw swap".into()),
            "a heartbeat timeout must not overwrite the operator's drain reason"
        );
        assert!(node.admin_locked);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mark_down_uses_a_liveness_reason_on_an_unlocked_node() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);

        cm.apply_health_actions(vec![super::HealthAction::MarkDown {
            name: "n1".into(),
            old_state: NodeState::Idle,
            admin_locked: false,
        }]);
        wait_for("down applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Down)
        });

        let node = cm.get_node("n1").unwrap();
        assert_eq!(node.state_reason, Some("Not responding".into()));
        assert!(!node.admin_locked);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_restores_allocated_state_rather_than_idle() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        if let Some(node) = cm.nodes.write().get_mut("n1") {
            node.state = NodeState::Down;
            node.alloc_resources.cpus = 4;
        }

        cm.apply_health_actions(vec![super::HealthAction::Recover {
            name: "n1".into(),
            old_state: NodeState::Down,
        }]);
        wait_for("recovery applied", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state != NodeState::Down)
        });
        assert_eq!(cm.get_node("n1").unwrap().state, NodeState::Allocated);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admin_drained_node_stays_locked_through_timeout_and_reregister() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "locked", 4, 8000);

        // Give the node an allocation so Drain becomes Draining
        let id = submit_and_wait(&cm, basic_spec("hold-job"));
        let alloc = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["locked".into()],
            alloc.clone(),
            per_node_for(&["locked"], alloc),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);

        // Admin drains while job is running — becomes Draining (admin_locked)
        cm.update_node_state("locked", NodeState::Drain, Some("hw swap".into()), None)
            .unwrap();
        wait_for("draining applied", || {
            cm.get_node("locked")
                .is_some_and(|n| n.state == NodeState::Draining)
        });
        assert!(cm.get_node("locked").unwrap().admin_locked);

        // Heartbeat times out — Draining → Down, admin_locked preserved
        if let Some(node) = cm.nodes.write().get_mut("locked") {
            node.last_heartbeat = Some(Utc::now() - chrono::Duration::seconds(200));
        }
        cm.check_node_health(90, super::MarkDownPolicy::Allowed);
        wait_for("health check applied", || {
            cm.get_node("locked")
                .is_some_and(|n| n.state == NodeState::Down)
        });
        let node = cm.get_node("locked").unwrap();
        assert_eq!(node.state, NodeState::Down);
        assert!(
            node.admin_locked,
            "admin lock must survive heartbeat timeout"
        );

        // Agent reconnects — re-registration must NOT recover to Idle
        cm.register_node(
            "locked".into(),
            "locked".into(),
            ResourceSet {
                cpus: 4,
                memory_mb: 8000,
                ..Default::default()
            },
            "127.0.0.1".into(),
            6818,
            String::new(),
            "1.0".into(),
            NodeSource::NativeHost,
            HashMap::new(),
            true,
        )
        .unwrap();
        let node = cm.get_node("locked").unwrap();
        assert_eq!(
            node.state,
            NodeState::Down,
            "admin-locked node must not auto-recover"
        );
        assert!(node.admin_locked);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn requeue_resets_fields_via_apply() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("requeue-me"));

        let alloc = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            alloc.clone(),
            per_node_for(&["n1"], alloc),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);

        // A reported path must not survive requeue (points at a node the job leaves).
        cm.set_job_output_paths(id, "/tmp/spur.out".into(), "/tmp/spur.out".into());
        assert!(cm.get_job(id).unwrap().actual_stdout_path.is_some());

        cm.apply_operation(&WalOperation::JobComplete {
            job_id: id,
            exit_code: -1,
            state: JobState::Timeout,
        });
        assert_eq!(cm.get_job(id).unwrap().state, JobState::Timeout);

        // Requeue: Timeout → Pending should reset allocation fields
        cm.apply_operation(&WalOperation::job_state_change(
            id,
            JobState::Timeout,
            JobState::Pending,
        ));

        let job = cm.get_job(id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.requeue_count, 1);
        assert!(
            job.start_time.is_none(),
            "start_time should be cleared on requeue"
        );
        assert!(
            job.allocated_nodes.is_empty(),
            "allocated_nodes should be cleared"
        );
        assert!(
            job.allocated_resources.is_none(),
            "allocated_resources should be cleared"
        );
        assert_eq!(job.pending_reason, PendingReason::None);
        assert!(
            job.actual_stdout_path.is_none() && job.actual_stderr_path.is_none(),
            "stale reported output path should be cleared on requeue"
        );
    }

    // Empty ("" from a mixed-version agent) must not shadow the computed fallback.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn set_job_output_paths_ignores_empty() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("empty-paths"));

        cm.set_job_output_paths(id, String::new(), String::new());
        let job = cm.get_job(id).unwrap();
        assert!(job.actual_stdout_path.is_none());
        assert!(job.actual_stderr_path.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn requeue_job_frees_node_resources() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("dispatch-fail"));

        let alloc = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into()],
            alloc.clone(),
            per_node_for(&["n1"], alloc),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);

        let node = cm.get_node("n1").unwrap();
        assert_eq!(
            node.alloc_resources.cpus, 2,
            "CPUs should be allocated after start"
        );

        // Simulate all-dispatch-failed requeue (the fix under test)
        cm.requeue_job(id).unwrap();
        settle(&cm, id, JobState::Pending);

        let job = cm.get_job(id).unwrap();
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.requeue_count, 1);
        assert!(job.start_time.is_none(), "start_time should be cleared");
        assert!(
            job.allocated_nodes.is_empty(),
            "allocated_nodes should be cleared"
        );
        assert!(
            job.allocated_resources.is_none(),
            "allocated_resources should be cleared"
        );

        let node = cm.get_node("n1").unwrap();
        assert_eq!(
            node.alloc_resources.cpus, 0,
            "node CPUs must be freed after requeue"
        );
        assert!(
            !node.alloc_resources.has_devices(),
            "node devices must be freed after requeue"
        );
        assert_eq!(node.state, NodeState::Idle, "node should return to Idle");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evict_job_transitions_partial_dispatch_failure_to_nodefail() {
        // A multi-node job where the dispatch RPC only reaches some of the
        // assigned nodes must not be left running forever waiting on
        // completions from a node that never launched it.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        register_node(&cm, "n2", 4, 8000);

        let mut spec = basic_spec("partial-dispatch-fail");
        spec.num_nodes = 2;
        let id = submit_and_wait(&cm, spec);

        let alloc = scalar_alloc(2, 4000);
        cm.start_job(
            id,
            vec!["n1".into(), "n2".into()],
            scalar_alloc(4, 8000),
            per_node_for(&["n1", "n2"], alloc),
        )
        .unwrap();
        settle(&cm, id, JobState::Running);

        // n1's dispatch succeeded, n2's never reached the agent.
        cm.evict_job(id).unwrap();
        settle(&cm, id, JobState::NodeFail);

        let job = cm.get_job(id).unwrap();
        assert_eq!(job.state, JobState::NodeFail);
        assert_eq!(
            job.pending_reason,
            PendingReason::JobLaunchFailure,
            "partial-dispatch eviction must report the job never fully launched, \
             not a mid-run node failure"
        );
        assert!(
            job.node_completions.is_empty(),
            "node_completions must be cleared so a stray late report can't reopen the job"
        );

        for name in ["n1", "n2"] {
            let node = cm.get_node(name).unwrap();
            assert_eq!(
                node.alloc_resources.cpus, 0,
                "allocation on {name} must be freed on eviction"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evict_job_does_not_double_deallocate_a_node_that_already_completed() {
        // Job A spans n1+n2 and shares n1 with unrelated Job B. n1 reports
        // A's completion (freeing A's slice, moving A to Completing) before
        // n2's dispatch failure is discovered and A is evicted. Evicting A
        // must not subtract A's already-freed n1 slice a second time — doing
        // so would incorrectly free capacity that actually still belongs to
        // B, letting the node be oversubscribed.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16_000);
        register_node(&cm, "n2", 4, 8000);

        let mut spec_a = basic_spec("job-a-shares-n1");
        spec_a.num_nodes = 2;
        let job_a = submit_and_wait(&cm, spec_a);
        cm.start_job(
            job_a,
            vec!["n1".into(), "n2".into()],
            scalar_alloc(4, 8000),
            per_node_for(&["n1", "n2"], scalar_alloc(2, 4000)),
        )
        .unwrap();
        settle(&cm, job_a, JobState::Running);

        let spec_b = basic_spec("job-b-shares-n1");
        let job_b = submit_and_wait(&cm, spec_b);
        cm.start_job(
            job_b,
            vec!["n1".into()],
            scalar_alloc(3, 3000),
            per_node_for(&["n1"], scalar_alloc(3, 3000)),
        )
        .unwrap();
        settle(&cm, job_b, JobState::Running);

        assert_eq!(
            cm.get_node("n1").unwrap().alloc_resources.cpus,
            5,
            "n1 should hold both A's (2) and B's (3) allocations"
        );

        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: job_a,
            node_name: "n1".into(),
            exit_code: 0,
            signal: 0,
        });
        settle(&cm, job_a, JobState::Completing);
        assert_eq!(
            cm.get_node("n1").unwrap().alloc_resources.cpus,
            3,
            "A's n1 slice must already be freed by its own completion report, \
             leaving only B's 3 cpus"
        );

        cm.evict_job(job_a).unwrap();
        settle(&cm, job_a, JobState::NodeFail);

        assert_eq!(
            cm.get_node("n1").unwrap().alloc_resources.cpus,
            3,
            "evicting A must not re-subtract its already-freed n1 slice and \
             clobber B's still-running allocation"
        );
        assert_eq!(
            cm.get_node("n2").unwrap().alloc_resources.cpus,
            0,
            "n2's slice must still be freed by the eviction"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn register_node_gets_partition_via_propose() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "test-node", 4, 8000);

        let node = cm.get_node("test-node").unwrap();
        assert!(!node.partitions.is_empty());
        assert_eq!(node.partitions[0], "default");
    }

    // --- Pure evaluate_node_health tests (no Raft needed) ---

    fn make_health_node(
        name: &str,
        state: NodeState,
        admin_locked: bool,
        last_hb: Option<chrono::DateTime<Utc>>,
    ) -> Node {
        let mut node = Node::new(name.into(), ResourceSet::default());
        node.state = state;
        node.admin_locked = admin_locked;
        node.last_heartbeat = last_hb;
        node
    }

    #[test]
    fn health_stale_idle_marks_down() {
        let node = make_health_node(
            "n1",
            NodeState::Idle,
            false,
            Some(Utc::now() - chrono::Duration::seconds(200)),
        );
        let actions = super::evaluate_node_health(&[&node], Utc::now(), 90);
        assert_eq!(
            actions,
            vec![super::HealthAction::MarkDown {
                name: "n1".into(),
                old_state: NodeState::Idle,
                admin_locked: false,
            }]
        );
    }

    #[test]
    fn recovered_allocated_node_stays_allocated() {
        let mut node = make_health_node(
            "n1",
            NodeState::Down,
            false,
            Some(Utc::now() - chrono::Duration::seconds(10)),
        );
        node.total_resources.cpus = 8;
        node.alloc_resources.cpus = 8;
        assert_eq!(
            super::recovered_node_state(Some(&node)),
            NodeState::Allocated
        );
    }

    #[test]
    fn recovered_missing_node_is_idle() {
        assert_eq!(super::recovered_node_state(None), NodeState::Idle);
    }

    #[test]
    fn recovered_non_down_admin_hold_is_preserved() {
        for state in [NodeState::Drain, NodeState::Draining, NodeState::Suspended] {
            let mut node = make_health_node("n1", state, true, None);
            node.total_resources.cpus = 8;
            assert_eq!(super::recovered_node_state(Some(&node)), state, "{state:?}");
        }
    }

    #[test]
    fn recovered_admin_locked_down_node_stays_down() {
        let node = make_health_node("n1", NodeState::Down, true, None);
        assert_eq!(super::recovered_node_state(Some(&node)), NodeState::Down);
    }

    #[test]
    fn recovered_unlocked_error_node_returns_to_idle() {
        let node = make_health_node("n1", NodeState::Error, false, None);
        assert_eq!(super::recovered_node_state(Some(&node)), NodeState::Idle);
    }

    #[test]
    fn leadership_grace_suppresses_until_the_window_elapses() {
        let grace = std::time::Duration::from_secs(90);
        let mut gate = super::LeadershipGrace::new(grace);
        let t0 = std::time::Instant::now();

        assert_eq!(
            gate.observe(true, t0),
            super::MarkDownPolicy::Suppressed,
            "leadership just acquired"
        );
        assert_eq!(
            gate.observe(true, t0 + std::time::Duration::from_secs(89)),
            super::MarkDownPolicy::Suppressed
        );
        assert_eq!(
            gate.observe(true, t0 + grace),
            super::MarkDownPolicy::Allowed
        );
    }

    #[test]
    fn leadership_grace_restarts_after_losing_leadership() {
        let grace = std::time::Duration::from_secs(90);
        let mut gate = super::LeadershipGrace::new(grace);
        let t0 = std::time::Instant::now();
        assert_eq!(gate.observe(true, t0), super::MarkDownPolicy::Suppressed);
        assert_eq!(
            gate.observe(true, t0 + grace),
            super::MarkDownPolicy::Allowed
        );

        let lost = t0 + std::time::Duration::from_secs(200);
        assert_eq!(gate.observe(false, lost), super::MarkDownPolicy::Suppressed);
        assert_eq!(
            gate.observe(true, lost + std::time::Duration::from_secs(1)),
            super::MarkDownPolicy::Suppressed,
            "re-acquiring leadership restarts the grace window"
        );
        assert_eq!(
            gate.observe(true, lost + std::time::Duration::from_secs(1) + grace),
            super::MarkDownPolicy::Allowed
        );
    }

    #[test]
    fn health_fresh_down_recovers() {
        let node = make_health_node(
            "n1",
            NodeState::Down,
            false,
            Some(Utc::now() - chrono::Duration::seconds(10)),
        );
        let actions = super::evaluate_node_health(&[&node], Utc::now(), 90);
        assert_eq!(
            actions,
            vec![super::HealthAction::Recover {
                name: "n1".into(),
                old_state: NodeState::Down,
            }]
        );
    }

    #[test]
    fn health_admin_locked_down_no_recovery() {
        let node = make_health_node(
            "n1",
            NodeState::Down,
            true,
            Some(Utc::now() - chrono::Duration::seconds(10)),
        );
        let actions = super::evaluate_node_health(&[&node], Utc::now(), 90);
        assert!(actions.is_empty());
    }

    #[test]
    fn health_drain_not_marked_down() {
        let node = make_health_node(
            "n1",
            NodeState::Drain,
            true,
            Some(Utc::now() - chrono::Duration::seconds(200)),
        );
        let actions = super::evaluate_node_health(&[&node], Utc::now(), 90);
        assert!(actions.is_empty());
    }

    #[test]
    fn health_idle_fresh_no_action() {
        let node = make_health_node(
            "n1",
            NodeState::Idle,
            false,
            Some(Utc::now() - chrono::Duration::seconds(10)),
        );
        let actions = super::evaluate_node_health(&[&node], Utc::now(), 90);
        assert!(actions.is_empty());
    }

    #[test]
    fn health_no_heartbeat_skipped() {
        let node = make_health_node("n1", NodeState::Idle, false, None);
        let actions = super::evaluate_node_health(&[&node], Utc::now(), 90);
        assert!(actions.is_empty());
    }

    #[test]
    fn health_mixed_actions() {
        let stale = make_health_node(
            "stale",
            NodeState::Idle,
            false,
            Some(Utc::now() - chrono::Duration::seconds(200)),
        );
        let recovering = make_health_node(
            "back",
            NodeState::Down,
            false,
            Some(Utc::now() - chrono::Duration::seconds(10)),
        );
        let stable = make_health_node(
            "ok",
            NodeState::Idle,
            false,
            Some(Utc::now() - chrono::Duration::seconds(10)),
        );
        let actions = super::evaluate_node_health(&[&stale, &recovering, &stable], Utc::now(), 90);
        assert_eq!(actions.len(), 2);
        assert_eq!(
            actions[0],
            super::HealthAction::MarkDown {
                name: "stale".into(),
                old_state: NodeState::Idle,
                admin_locked: false,
            }
        );
        assert_eq!(
            actions[1],
            super::HealthAction::Recover {
                name: "back".into(),
                old_state: NodeState::Down,
            }
        );
    }

    // --- Pure evaluate_registration tests ---

    #[test]
    fn registration_new_node() {
        let resources = ResourceSet {
            cpus: 4,
            memory_mb: 8000,
            ..Default::default()
        };
        assert_eq!(
            super::evaluate_registration(None, &resources),
            super::RegistrationAction::Register,
        );
    }

    #[test]
    fn registration_unchanged_skip() {
        let resources = ResourceSet {
            cpus: 4,
            memory_mb: 8000,
            ..Default::default()
        };
        let node = Node::new("n1".into(), resources.clone());
        assert_eq!(
            super::evaluate_registration(Some(&node), &resources),
            super::RegistrationAction::Skip,
        );
    }

    #[test]
    fn registration_resources_changed_update() {
        let old = ResourceSet {
            cpus: 4,
            memory_mb: 8000,
            ..Default::default()
        };
        let new = ResourceSet {
            cpus: 8,
            memory_mb: 16000,
            ..Default::default()
        };
        let node = Node::new("n1".into(), old);
        assert_eq!(
            super::evaluate_registration(Some(&node), &new),
            super::RegistrationAction::Update,
        );
    }

    // --- expand_job_specs tests ---

    #[test]
    fn expand_non_array_returns_single_spec() {
        let spec = basic_spec("simple");
        let result = super::expand_job_specs(spec, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "simple");
        assert!(result[0].array_job_id.is_none());
        assert!(result[0].array_task_id.is_none());
        assert!(result[0].array_max_concurrent.is_none());
    }

    #[test]
    fn expand_array_with_throttle() {
        let mut spec = basic_spec("arr");
        spec.array_spec = Some("0-4%2".into());
        let result = super::expand_job_specs(spec, 10).unwrap();
        assert_eq!(result.len(), 5);
        for (i, s) in result.iter().enumerate() {
            assert_eq!(s.array_job_id, Some(10));
            assert_eq!(s.array_task_id, Some(i as u32));
            assert_eq!(s.array_max_concurrent, Some(2));
            assert!(s.array_spec.is_none());
            assert_eq!(s.name, "arr");
        }
    }

    #[test]
    fn expand_array_without_throttle() {
        let mut spec = basic_spec("arr");
        spec.array_spec = Some("0-4".into());
        let result = super::expand_job_specs(spec, 5).unwrap();
        assert_eq!(result.len(), 5);
        for s in &result {
            assert_eq!(s.array_job_id, Some(5));
            assert!(s.array_max_concurrent.is_none());
        }
    }

    #[test]
    fn expand_array_invalid_spec_errors() {
        let mut spec = basic_spec("bad");
        spec.array_spec = Some("10-5".into());
        assert!(super::expand_job_specs(spec, 1).is_err());
    }

    // --- apply_default_partition tests ---

    #[test]
    fn apply_default_partition_picks_default() {
        let mut spec = basic_spec("j");
        spec.partition = None;
        let partitions = vec![
            Partition {
                name: "other".into(),
                is_default: false,
                ..Default::default()
            },
            Partition {
                name: "gpu".into(),
                is_default: true,
                ..Default::default()
            },
        ];
        super::apply_default_partition(&mut spec, &partitions);
        assert_eq!(spec.partition.as_deref(), Some("gpu"));
    }

    /// Defaulting with no QOS or account wall ceiling in play.
    fn default_time_limit(spec: &mut JobSpec, partitions: &[Partition], cluster_minutes: u32) {
        super::apply_default_time_limit(spec, partitions, cluster_minutes, WallCaps::default());
    }

    #[test]
    fn apply_default_time_limit_uses_partition_default() {
        let mut spec = basic_spec("j");
        spec.partition = Some("gpu".into());
        spec.time_limit = None;
        let partitions = vec![Partition {
            name: "gpu".into(),
            default_time_minutes: Some(30),
            ..Default::default()
        }];
        default_time_limit(&mut spec, &partitions, 0);
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(30)));
    }

    #[test]
    fn apply_default_time_limit_noop_when_set() {
        let mut spec = basic_spec("j");
        spec.time_limit = Some(chrono::Duration::minutes(5));
        let partitions = vec![Partition {
            name: "gpu".into(),
            default_time_minutes: Some(30),
            ..Default::default()
        }];
        default_time_limit(&mut spec, &partitions, 0);
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(5)));
    }

    #[test]
    fn apply_default_time_limit_skips_when_partition_has_no_default() {
        let mut spec = basic_spec("j");
        spec.partition = Some("gpu".into());
        spec.time_limit = None;
        let partitions = vec![Partition {
            name: "gpu".into(),
            default_time_minutes: None,
            ..Default::default()
        }];
        default_time_limit(&mut spec, &partitions, 0);
        assert!(spec.time_limit.is_none());
    }

    #[test]
    fn apply_default_time_limit_falls_back_to_partition_max_time() {
        // No DefaultTime but a finite MaxTime: Slurm uses MaxTime as the default.
        let mut spec = basic_spec("j");
        spec.partition = Some("gpu".into());
        spec.time_limit = None;
        let partitions = vec![Partition {
            name: "gpu".into(),
            default_time_minutes: None,
            max_time_minutes: Some(120),
            ..Default::default()
        }];
        default_time_limit(&mut spec, &partitions, 60);
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(120)));
    }

    #[test]
    fn apply_default_time_limit_falls_back_to_cluster_default_on_unlimited() {
        // UNLIMITED partition (no DefaultTime, no MaxTime): the cluster-wide
        // default bounds the job so it cannot run forever.
        let mut spec = basic_spec("j");
        spec.partition = Some("gpu".into());
        spec.time_limit = None;
        let partitions = vec![Partition {
            name: "gpu".into(),
            default_time_minutes: None,
            max_time_minutes: None,
            ..Default::default()
        }];
        default_time_limit(&mut spec, &partitions, 60);
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(60)));
    }

    #[test]
    fn apply_default_time_limit_unlimited_when_cluster_default_disabled() {
        // cluster default 0 = disabled: an UNLIMITED partition leaves the job
        // with no wall-time (opt-out escape hatch).
        let mut spec = basic_spec("j");
        spec.partition = Some("gpu".into());
        spec.time_limit = None;
        let partitions = vec![Partition {
            name: "gpu".into(),
            default_time_minutes: None,
            max_time_minutes: None,
            ..Default::default()
        }];
        default_time_limit(&mut spec, &partitions, 0);
        assert!(spec.time_limit.is_none());
    }

    #[test]
    fn apply_default_time_limit_multi_partition_uses_min_across_requested() {
        // A `-t`-less job to two partitions gets the smaller MaxTime so the
        // default fits both (and would pass EnforcePartLimits=ALL).
        let mut spec = basic_spec("j");
        spec.partition = Some("big,small".into());
        spec.time_limit = None;
        let partitions = vec![
            Partition {
                name: "big".into(),
                default_time_minutes: None,
                max_time_minutes: Some(120),
                ..Default::default()
            },
            Partition {
                name: "small".into(),
                default_time_minutes: None,
                max_time_minutes: Some(60),
                ..Default::default()
            },
        ];
        default_time_limit(&mut spec, &partitions, 60);
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(60)));
    }

    #[test]
    fn apply_default_time_limit_unbounded_partition_leaves_job_unbounded() {
        // With the cluster fallback disabled, a requested partition that resolves
        // to no bound keeps the whole job unbounded even if a sibling is bounded.
        let mut spec = basic_spec("j");
        spec.partition = Some("bounded,unlimited".into());
        spec.time_limit = None;
        let partitions = vec![
            Partition {
                name: "bounded".into(),
                default_time_minutes: Some(30),
                ..Default::default()
            },
            Partition {
                name: "unlimited".into(),
                default_time_minutes: None,
                max_time_minutes: None,
                ..Default::default()
            },
        ];
        default_time_limit(&mut spec, &partitions, 0);
        assert!(spec.time_limit.is_none());
    }

    #[test]
    fn apply_default_time_limit_prefers_default_over_smaller_max_time() {
        // DefaultTime wins the fallback chain even when MaxTime is smaller and a
        // cluster default is set.
        let mut spec = basic_spec("j");
        spec.partition = Some("gpu".into());
        spec.time_limit = None;
        let partitions = vec![Partition {
            name: "gpu".into(),
            default_time_minutes: Some(45),
            max_time_minutes: Some(30),
            ..Default::default()
        }];
        default_time_limit(&mut spec, &partitions, 60);
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(45)));
    }

    #[test]
    fn apply_default_time_limit_multi_partition_uses_min_across_default_times() {
        // Two partitions each with a distinct DefaultTime: the smaller wins so
        // the auto-default fits both.
        let mut spec = basic_spec("j");
        spec.partition = Some("a,b".into());
        spec.time_limit = None;
        let partitions = vec![
            Partition {
                name: "a".into(),
                default_time_minutes: Some(45),
                ..Default::default()
            },
            Partition {
                name: "b".into(),
                default_time_minutes: Some(20),
                ..Default::default()
            },
        ];
        default_time_limit(&mut spec, &partitions, 0);
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(20)));
    }

    /// A job that asked for no wall-time, in a partition that imposes no default.
    fn unbounded_spec_and_partition() -> (JobSpec, Vec<Partition>) {
        let mut spec = basic_spec("j");
        spec.partition = Some("gpu".into());
        spec.time_limit = None;
        let partitions = vec![Partition {
            name: "gpu".into(),
            ..Default::default()
        }];
        (spec, partitions)
    }

    fn qos_wall(minutes: u32) -> WallCaps {
        WallCaps {
            qos_minutes: Some(minutes),
            account_minutes: None,
        }
    }

    // No partition in play means nothing caps the wall ceiling. Folding over an
    // empty slice from a zero seed would instead collapse the cap to a zero-length
    // limit, so the ceiling is computed without relying on a non-empty caller.
    #[test]
    fn wall_cap_default_without_partitions_keeps_the_cap() {
        assert_eq!(super::wall_cap_default(&[], qos_wall(720)), Some(720));
    }

    #[test]
    fn apply_default_time_limit_uses_qos_max_wall_when_partitions_impose_none() {
        let (mut spec, partitions) = unbounded_spec_and_partition();
        super::apply_default_time_limit(&mut spec, &partitions, 0, qos_wall(720));
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(720)));
    }

    #[test]
    fn apply_default_time_limit_leaves_a_requested_limit_below_max_wall_alone() {
        let (mut spec, partitions) = unbounded_spec_and_partition();
        spec.time_limit = Some(chrono::Duration::minutes(30));
        super::apply_default_time_limit(&mut spec, &partitions, 0, qos_wall(720));
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(30)));
    }

    #[test]
    fn apply_default_time_limit_prefers_the_partition_default_over_max_wall() {
        // MaxWall is a ceiling, not a default: a partition that supplies one keeps it.
        let mut spec = basic_spec("j");
        spec.partition = Some("gpu".into());
        spec.time_limit = None;
        let partitions = vec![Partition {
            name: "gpu".into(),
            default_time_minutes: Some(30),
            ..Default::default()
        }];
        super::apply_default_time_limit(&mut spec, &partitions, 0, qos_wall(720));
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(30)));
    }

    #[test]
    fn apply_default_time_limit_stays_unbounded_without_any_wall_cap() {
        let (mut spec, partitions) = unbounded_spec_and_partition();
        super::apply_default_time_limit(&mut spec, &partitions, 0, WallCaps::default());
        assert!(spec.time_limit.is_none());
    }

    #[test]
    fn apply_default_time_limit_ignores_a_zero_max_wall() {
        // `0` is the block-all sentinel; defaulting to it would submit a job that
        // its own limit kills on the first watchdog tick.
        let (mut spec, partitions) = unbounded_spec_and_partition();
        super::apply_default_time_limit(&mut spec, &partitions, 0, qos_wall(0));
        assert!(spec.time_limit.is_none());
    }

    #[test]
    fn apply_default_time_limit_holds_max_wall_under_the_account_wall() {
        // An account MaxWall rejects the submission outright, so a default above it
        // would turn a job that runs today into one that cannot be submitted.
        let (mut spec, partitions) = unbounded_spec_and_partition();
        let caps = WallCaps {
            qos_minutes: Some(720),
            account_minutes: Some(360),
        };
        super::apply_default_time_limit(&mut spec, &partitions, 0, caps);
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(360)));
    }

    #[test]
    fn apply_default_time_limit_uses_the_account_wall_when_the_qos_has_none() {
        let (mut spec, partitions) = unbounded_spec_and_partition();
        let caps = WallCaps {
            qos_minutes: None,
            account_minutes: Some(90),
        };
        super::apply_default_time_limit(&mut spec, &partitions, 0, caps);
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(90)));
    }

    #[test]
    fn apply_default_time_limit_ignores_a_zero_account_wall() {
        let (mut spec, partitions) = unbounded_spec_and_partition();
        let caps = WallCaps {
            qos_minutes: Some(720),
            account_minutes: Some(0),
        };
        super::apply_default_time_limit(&mut spec, &partitions, 0, caps);
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(720)));
    }

    #[test]
    fn apply_default_time_limit_holds_max_wall_under_the_partition_max_time() {
        // Above MaxTime the job would pend forever on PartitionTimeLimit, for a
        // limit it never asked for.
        let mut spec = basic_spec("j");
        spec.partition = Some("gpu".into());
        spec.time_limit = None;
        let partitions = vec![Partition {
            name: "gpu".into(),
            max_time_minutes: Some(240),
            ..Default::default()
        }];
        super::apply_default_time_limit(&mut spec, &partitions, 0, qos_wall(720));
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(240)));
    }

    #[test]
    fn apply_default_time_limit_takes_the_most_permissive_partition_max_time() {
        // The job only has to be schedulable somewhere, so the widest requested
        // MaxTime is the ceiling — not the narrowest.
        let mut spec = basic_spec("j");
        spec.partition = Some("small,big".into());
        spec.time_limit = None;
        let partitions = vec![
            Partition {
                name: "small".into(),
                max_time_minutes: Some(60),
                ..Default::default()
            },
            Partition {
                name: "big".into(),
                max_time_minutes: Some(240),
                ..Default::default()
            },
        ];
        super::apply_default_time_limit(&mut spec, &partitions, 0, qos_wall(720));
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(240)));
    }

    #[test]
    fn apply_default_time_limit_ignores_partition_max_time_when_one_is_unlimited() {
        let mut spec = basic_spec("j");
        spec.partition = Some("capped,unlimited".into());
        spec.time_limit = None;
        let partitions = vec![
            Partition {
                name: "capped".into(),
                max_time_minutes: Some(60),
                ..Default::default()
            },
            Partition {
                name: "unlimited".into(),
                ..Default::default()
            },
        ];
        super::apply_default_time_limit(&mut spec, &partitions, 0, qos_wall(720));
        assert_eq!(spec.time_limit, Some(chrono::Duration::minutes(720)));
    }

    #[test]
    fn wall_caps_reports_nothing_until_the_caches_load() {
        // Fail-open: an unread limit must not decide a job's deadline.
        let dir = TempDir::new().unwrap();
        let cluster = ClusterManager::new(test_config(), dir.path()).unwrap();
        let mut spec = basic_spec("j");
        spec.qos = Some("burst".into());
        spec.account = Some("eng".into());

        let caps = cluster.wall_caps(&spec);
        assert!(caps.qos_minutes.is_none());
        assert!(caps.account_minutes.is_none());
    }

    #[test]
    fn validate_partition_time_limit_enforces_down_partitions_under_all() {
        // Slurm parity: under ALL, a requested Down partition's stricter MaxTime
        // still rejects, regardless of its state (deterministic at submit).
        let mut spec = basic_spec("j");
        spec.partition = Some("up_big,down_small".into());
        spec.time_limit = Some(chrono::Duration::minutes(90));
        let partitions = vec![
            Partition {
                name: "up_big".into(),
                state: spur_core::partition::PartitionState::Up,
                max_time_minutes: Some(120),
                ..Default::default()
            },
            Partition {
                name: "down_small".into(),
                state: spur_core::partition::PartitionState::Down,
                max_time_minutes: Some(10),
                ..Default::default()
            },
        ];
        assert!(super::validate_partition_time_limit(
            &spec,
            spur_core::config::EnforcePartLimits::All,
            &partitions,
        )
        .is_err());
    }

    #[test]
    fn validate_partition_time_limit_enforces_inactive_only_request() {
        // A request naming only a Down partition is still checked (not silently
        // admitted): an over-limit `-t` is rejected, a within-limit one passes.
        let partitions = vec![Partition {
            name: "down_small".into(),
            state: spur_core::partition::PartitionState::Down,
            max_time_minutes: Some(10),
            ..Default::default()
        }];

        let mut over = basic_spec("over");
        over.partition = Some("down_small".into());
        over.time_limit = Some(chrono::Duration::minutes(90));
        assert!(super::validate_partition_time_limit(
            &over,
            spur_core::config::EnforcePartLimits::All,
            &partitions,
        )
        .is_err());

        let mut within = basic_spec("within");
        within.partition = Some("down_small".into());
        within.time_limit = Some(chrono::Duration::minutes(5));
        assert!(super::validate_partition_time_limit(
            &within,
            spur_core::config::EnforcePartLimits::All,
            &partitions,
        )
        .is_ok());
    }

    #[test]
    fn validate_partition_time_limit_rejects_over_limit_up_partition() {
        let mut spec = basic_spec("j");
        spec.partition = Some("up_small".into());
        spec.time_limit = Some(chrono::Duration::minutes(90));
        let partitions = vec![Partition {
            name: "up_small".into(),
            state: spur_core::partition::PartitionState::Up,
            max_time_minutes: Some(10),
            ..Default::default()
        }];
        assert!(super::validate_partition_time_limit(
            &spec,
            spur_core::config::EnforcePartLimits::All,
            &partitions,
        )
        .is_err());
    }

    #[test]
    fn apply_default_partition_falls_back_to_first() {
        let mut spec = basic_spec("j");
        spec.partition = None;
        let partitions = vec![Partition {
            name: "batch".into(),
            is_default: false,
            ..Default::default()
        }];
        super::apply_default_partition(&mut spec, &partitions);
        assert_eq!(spec.partition.as_deref(), Some("batch"));
    }

    #[test]
    fn apply_default_partition_noop_when_set() {
        let mut spec = basic_spec("j");
        spec.partition = Some("mypart".into());
        let partitions = vec![Partition {
            name: "default".into(),
            is_default: true,
            ..Default::default()
        }];
        super::apply_default_partition(&mut spec, &partitions);
        assert_eq!(spec.partition.as_deref(), Some("mypart"));
    }

    #[test]
    fn apply_default_partition_treats_empty_string_as_unset() {
        let mut spec = basic_spec("j");
        spec.partition = Some(String::new());
        let partitions = vec![Partition {
            name: "gpu".into(),
            is_default: true,
            ..Default::default()
        }];
        super::apply_default_partition(&mut spec, &partitions);
        assert_eq!(spec.partition.as_deref(), Some("gpu"));
    }

    // --- apply_default_qos tests ---

    fn qos_cache_with(names: &[&str]) -> QosCache {
        let cache = QosCache::new();
        for name in names {
            cache.insert(Qos {
                name: (*name).into(),
                ..Default::default()
            });
        }
        cache
    }

    // Inert config: no fallback, require_qos off (base resolution chain).
    fn acct_cfg() -> spur_core::config::AccountingConfig {
        spur_core::config::AccountingConfig::default()
    }

    fn acct_cfg_with(default_qos: &str, require_qos: bool) -> spur_core::config::AccountingConfig {
        spur_core::config::AccountingConfig {
            default_qos: default_qos.into(),
            require_qos,
            ..Default::default()
        }
    }

    #[test]
    fn apply_default_qos_explicit_valid_passes_through() {
        let assoc = AssociationCache::new();
        let qos = qos_cache_with(&["highprio"]);
        let mut spec = basic_spec("j");
        spec.qos = Some("highprio".into());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap();
        assert_eq!(spec.qos.as_deref(), Some("highprio"));
    }

    #[test]
    fn apply_default_qos_explicit_invalid_is_rejected() {
        let assoc = AssociationCache::new();
        let qos = qos_cache_with(&["normal"]);
        let mut spec = basic_spec("j");
        spec.qos = Some("doesnotexist".into());

        let err = super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("QOS 'doesnotexist' does not exist")
        );
        assert!(!err.to_string().contains("accounting may not be enabled"));
    }

    #[test]
    fn apply_default_qos_explicit_invalid_hints_when_cache_unloaded() {
        let assoc = AssociationCache::new();
        let qos = QosCache::new();
        let mut spec = basic_spec("j");
        spec.qos = Some("any-qos".into());

        let err = super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap_err();
        assert!(err.to_string().contains("QOS 'any-qos' does not exist"));
        assert!(err.to_string().contains("accounting may not be enabled"));
    }

    #[test]
    fn apply_default_qos_inherits_association_default_with_explicit_account() {
        let assoc = AssociationCache::new();
        assoc.insert_default_qos("testuser", "research", "highprio");
        let qos = qos_cache_with(&["highprio"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap();
        assert_eq!(spec.qos.as_deref(), Some("highprio"));
    }

    #[test]
    fn apply_default_account_inherits_users_default_when_unset() {
        let assoc = AssociationCache::new();
        assoc.insert_default_account("testuser", "research");
        let mut spec = basic_spec("j");
        assert!(spec.account.is_none());

        super::apply_default_account(&mut spec, &assoc);
        assert_eq!(spec.account.as_deref(), Some("research"));
    }

    #[test]
    fn apply_default_account_skips_when_cache_not_loaded() {
        let assoc = AssociationCache::new();
        let mut spec = basic_spec("j");
        super::apply_default_account(&mut spec, &assoc);
        assert!(spec.account.is_none());
    }

    #[test]
    fn apply_default_account_noop_when_explicit() {
        let assoc = AssociationCache::new();
        assoc.insert_default_account("testuser", "research");
        let mut spec = basic_spec("j");
        spec.account = Some("faculty".into());

        super::apply_default_account(&mut spec, &assoc);
        assert_eq!(spec.account.as_deref(), Some("faculty"));
    }

    // --- validate_user_account / require_association tests ---

    fn acct_cfg_with_require_association(
        require_association: bool,
    ) -> spur_core::config::AccountingConfig {
        spur_core::config::AccountingConfig {
            require_association,
            ..Default::default()
        }
    }

    #[test]
    fn validate_user_account_allows_no_account_when_require_association_off() {
        let assoc = AssociationCache::new();
        let spec = basic_spec("j");
        assert!(spec.account.is_none());

        super::validate_user_account(&spec, &assoc, &acct_cfg_with_require_association(false))
            .unwrap();
    }

    #[test]
    fn validate_user_account_rejects_no_account_when_require_association_on() {
        let assoc = AssociationCache::new();
        assoc.insert_association("someone-else", "research"); // loads the cache
        let spec = basic_spec("j");
        assert!(spec.account.is_none());

        let err =
            super::validate_user_account(&spec, &assoc, &acct_cfg_with_require_association(true))
                .unwrap_err();
        assert!(err
            .to_string()
            .contains("no account resolved for user 'testuser'"));
    }

    #[test]
    fn validate_user_account_rejects_no_account_even_when_user_has_associations_but_no_default() {
        // A real reachable state: an association exists but none is flagged default.
        let assoc = AssociationCache::new();
        assoc.insert_association("testuser", "research");
        let spec = basic_spec("j");
        assert!(spec.account.is_none());

        let err =
            super::validate_user_account(&spec, &assoc, &acct_cfg_with_require_association(true))
                .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no account resolved for user 'testuser'"));
        assert!(!msg.contains("has no account associations"));
    }

    #[test]
    fn validate_user_account_allows_resolved_default_account_when_require_association_on() {
        let assoc = AssociationCache::new();
        assoc.insert_default_account("testuser", "research");
        assoc.insert_association("testuser", "research");
        let mut spec = basic_spec("j");
        super::apply_default_account(&mut spec, &assoc);
        assert_eq!(spec.account.as_deref(), Some("research"));

        super::validate_user_account(&spec, &assoc, &acct_cfg_with_require_association(true))
            .unwrap();
    }

    #[test]
    fn validate_user_account_rejects_no_account_even_when_cache_unloaded() {
        // Unconditional like `require_qos`'s final check, but hints at the cold cache.
        let assoc = AssociationCache::new();
        let spec = basic_spec("j");
        assert!(!assoc.is_loaded());

        let err =
            super::validate_user_account(&spec, &assoc, &acct_cfg_with_require_association(true))
                .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no account resolved for user 'testuser'"));
        assert!(msg.contains("accounting may not be enabled"));
    }

    #[test]
    fn validate_user_account_still_rejects_bad_explicit_account_when_require_association_on() {
        let assoc = AssociationCache::new();
        assoc.insert_association("testuser", "research");
        let mut spec = basic_spec("j");
        spec.account = Some("other".into());

        let err =
            super::validate_user_account(&spec, &assoc, &acct_cfg_with_require_association(true))
                .unwrap_err();
        assert!(err
            .to_string()
            .contains("is not associated with account 'other'"));
    }

    #[test]
    fn validate_user_account_fails_closed_on_cold_cache_with_explicit_account_and_require_association(
    ) {
        // require_association=true doesn't bypass the cold-cache fence for an explicit
        // account. The fence is transient (cache still loading), so it must be retryable.
        let assoc = AssociationCache::new(); // never loaded
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());
        let cfg = spur_core::config::AccountingConfig {
            database_url: "postgresql://test".into(),
            require_association: true,
            ..Default::default()
        };

        let err = super::validate_user_account(&spec, &assoc, &cfg).unwrap_err();
        assert!(
            matches!(&err, super::SubmitError::Unavailable(m) if m.contains("temporarily unavailable")),
            "cold cache must be retryable, got {err:?}"
        );
    }

    #[test]
    fn validate_user_account_cold_cache_no_account_require_association_is_retryable() {
        // A user's default account may still be loading; rejecting it permanently would
        // fail a job that would succeed once the refresh loop populates the cache.
        let assoc = AssociationCache::new(); // never loaded
        let spec = basic_spec("j");
        assert!(spec.account.is_none());
        let cfg = spur_core::config::AccountingConfig {
            database_url: "postgresql://test".into(),
            require_association: true,
            ..Default::default()
        };

        let err = super::validate_user_account(&spec, &assoc, &cfg).unwrap_err();
        assert!(
            matches!(&err, super::SubmitError::Unavailable(m) if m.contains("temporarily unavailable")),
            "cold cache must be retryable, got {err:?}"
        );
    }

    #[test]
    fn apply_default_qos_inherits_via_users_default_account_when_no_dash_a() {
        let assoc = AssociationCache::new();
        assoc.insert_default_account("testuser", "research");
        assoc.insert_default_qos("testuser", "research", "highprio");
        let qos = qos_cache_with(&["highprio"]);
        let mut spec = basic_spec("j");
        // No --account given at all.
        assert!(spec.account.is_none());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap();
        assert_eq!(spec.qos.as_deref(), Some("highprio"));
    }

    #[test]
    fn apply_default_qos_no_association_default_leaves_qos_unset() {
        let assoc = AssociationCache::new();
        let qos = QosCache::new();
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap();
        assert_eq!(spec.qos, None);
    }

    #[test]
    fn apply_default_qos_stale_association_default_degrades_silently() {
        let assoc = AssociationCache::new();
        // Association still points at a QOS that has since been deleted.
        assoc.insert_default_qos("testuser", "research", "deleted-qos");
        let qos = QosCache::new(); // empty: "deleted-qos" is not there
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap();
        assert_eq!(spec.qos, None, "must not fail submission on stale data");
    }

    #[test]
    fn apply_default_qos_falls_back_to_cluster_default() {
        // No --qos and no association default → cluster fallback applies.
        let assoc = AssociationCache::new();
        let qos = qos_cache_with(&["normal"]);
        let mut spec = basic_spec("j");

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg_with("normal", false)).unwrap();
        assert_eq!(spec.qos.as_deref(), Some("normal"));
    }

    #[test]
    fn apply_default_qos_explicit_permitted_despite_stale_association_default() {
        // Association's pinned default was deleted from the cluster; that must
        // not poison authorization for an unrelated, unrestricted explicit --qos.
        let assoc = AssociationCache::new();
        assoc.insert_default_qos("testuser", "research", "deleted-qos");
        let qos = qos_cache_with(&["highprio"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());
        spec.qos = Some("highprio".into());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap();
        assert_eq!(spec.qos.as_deref(), Some("highprio"));
    }

    #[test]
    fn apply_default_qos_cluster_fallback_ignored_outside_allow_list() {
        // Association is restricted to an allow-list that omits the cluster
        // fallback; omitting --qos must not silently grant it.
        let assoc = AssociationCache::new();
        assoc.insert_allowed_qos("testuser", "research", &["a", "b"]);
        let qos = qos_cache_with(&["a", "b", "cluster-default"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());

        super::apply_default_qos(
            &mut spec,
            &assoc,
            &qos,
            &acct_cfg_with("cluster-default", false),
        )
        .unwrap();
        assert_eq!(spec.qos, None, "unauthorized fallback must not be assigned");
    }

    #[test]
    fn apply_default_qos_cluster_fallback_outside_allow_list_with_require_qos_errors() {
        let assoc = AssociationCache::new();
        assoc.insert_allowed_qos("testuser", "research", &["a", "b"]);
        let qos = qos_cache_with(&["a", "b", "cluster-default"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());

        let err = super::apply_default_qos(
            &mut spec,
            &assoc,
            &qos,
            &acct_cfg_with("cluster-default", true),
        )
        .unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid(
                "no QOS specified and no default QOS is configured for this user/account"
            )
        );
    }

    #[test]
    fn apply_default_qos_cluster_fallback_permitted_when_in_allow_list() {
        let assoc = AssociationCache::new();
        assoc.insert_allowed_qos("testuser", "research", &["cluster-default"]);
        let qos = qos_cache_with(&["cluster-default"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());

        super::apply_default_qos(
            &mut spec,
            &assoc,
            &qos,
            &acct_cfg_with("cluster-default", false),
        )
        .unwrap();
        assert_eq!(spec.qos.as_deref(), Some("cluster-default"));
    }

    #[test]
    fn apply_default_qos_association_default_beats_cluster_default() {
        let assoc = AssociationCache::new();
        assoc.insert_default_qos("testuser", "research", "highprio");
        let qos = qos_cache_with(&["highprio", "normal"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg_with("normal", false)).unwrap();
        assert_eq!(
            spec.qos.as_deref(),
            Some("highprio"),
            "association default takes precedence over the cluster fallback"
        );
    }

    #[test]
    fn apply_default_qos_nonexistent_cluster_default_is_rejected() {
        // A misconfigured fallback must hard-error, not silently leave it unenforced.
        let assoc = AssociationCache::new();
        let qos = qos_cache_with(&["normal"]);
        let mut spec = basic_spec("j");

        let err = super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg_with("ghost", false))
            .unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid("configured default QOS 'ghost' does not exist")
        );
        assert!(!err.to_string().contains("accounting may not be enabled"));
    }

    #[test]
    fn apply_default_qos_cluster_default_hints_when_cache_unloaded() {
        let assoc = AssociationCache::new();
        let qos = QosCache::new();
        let mut spec = basic_spec("j");

        let err = super::apply_default_qos(
            &mut spec,
            &assoc,
            &qos,
            &acct_cfg_with("fallback-qos", false),
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("configured default QOS 'fallback-qos' does not exist"));
        assert!(err.to_string().contains("accounting may not be enabled"));
    }

    #[test]
    fn apply_default_qos_require_qos_rejects_when_none_resolves() {
        // require_qos with no fallback rejects a job that resolves to no QOS.
        let assoc = AssociationCache::new();
        let qos = QosCache::new();
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());

        let err = super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg_with("", true))
            .unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid(
                "no QOS specified and no default QOS is configured for this user/account"
            )
        );
    }

    #[test]
    fn apply_default_qos_require_qos_satisfied_by_cluster_default() {
        // With both set, the fallback satisfies require_qos — no rejection.
        let assoc = AssociationCache::new();
        let qos = qos_cache_with(&["normal"]);
        let mut spec = basic_spec("j");

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg_with("normal", true)).unwrap();
        assert_eq!(spec.qos.as_deref(), Some("normal"));
    }

    #[test]
    fn apply_default_qos_require_qos_satisfied_by_explicit() {
        // An explicit valid QOS satisfies require_qos regardless of fallback.
        let assoc = AssociationCache::new();
        let qos = qos_cache_with(&["highprio"]);
        let mut spec = basic_spec("j");
        spec.qos = Some("highprio".into());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg_with("", true)).unwrap();
        assert_eq!(spec.qos.as_deref(), Some("highprio"));
    }

    #[test]
    fn apply_default_qos_explicit_unauthorized_qos_is_rejected() {
        // An explicit QOS that exists but isn't the association's own default
        // must be rejected, not silently accepted (SPUR-101).
        let assoc = AssociationCache::new();
        assoc.insert_default_qos("testuser", "research", "highprio");
        let qos = qos_cache_with(&["highprio", "other-teams-qos"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());
        spec.qos = Some("other-teams-qos".into());

        let err = super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid(
                "QOS 'other-teams-qos' is not permitted for user 'testuser' under account 'research'"
            )
        );
    }

    #[test]
    fn apply_default_qos_explicit_matches_association_default_is_allowed() {
        let assoc = AssociationCache::new();
        assoc.insert_default_qos("testuser", "research", "highprio");
        let qos = qos_cache_with(&["highprio"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());
        spec.qos = Some("highprio".into());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap();
        assert_eq!(spec.qos.as_deref(), Some("highprio"));
    }

    #[test]
    fn apply_default_qos_explicit_allowed_when_association_has_no_default_pinned() {
        // A loaded cache with a real membership but no default QOS on record
        // for this association has never been pinned to one — stay permissive.
        let assoc = AssociationCache::new();
        assoc.insert_association("testuser", "research");
        let qos = qos_cache_with(&["anything"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());
        spec.qos = Some("anything".into());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap();
        assert_eq!(spec.qos.as_deref(), Some("anything"));
    }

    #[test]
    fn apply_default_qos_explicit_skips_authz_when_no_account_resolves() {
        // Cache loaded but the user has no resolvable account (e.g. no
        // default account and none given) — nothing to authorize against.
        let assoc = AssociationCache::new();
        assoc.set_loaded_without_associations();
        let qos = qos_cache_with(&["anything"]);
        let mut spec = basic_spec("j");
        spec.qos = Some("anything".into());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap();
        assert_eq!(spec.qos.as_deref(), Some("anything"));
    }

    #[test]
    fn apply_default_qos_explicit_rejects_other_accounts_qos() {
        // A user in two accounts, each pinned to its own QOS, must not be
        // able to borrow one account's QOS while submitting under the
        // other — the exact cross-account confusion reported in SPUR-101.
        let assoc = AssociationCache::new();
        assoc.insert_default_qos("testuser", "hyperloom", "hyperloom-qos");
        assoc.insert_default_qos("testuser", "primus", "primus-qos");
        let qos = qos_cache_with(&["hyperloom-qos", "primus-qos"]);
        let mut spec = basic_spec("j");
        spec.account = Some("hyperloom".into());
        spec.qos = Some("primus-qos".into());

        let err = super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid(
                "QOS 'primus-qos' is not permitted for user 'testuser' under account 'hyperloom'"
            )
        );
    }

    #[test]
    fn apply_default_qos_explicit_allows_any_member_of_the_allow_list() {
        let assoc = AssociationCache::new();
        assoc.insert_allowed_qos("testuser", "research", &["a", "b", "c"]);
        let qos = qos_cache_with(&["a", "b", "c"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());

        for name in ["a", "b", "c"] {
            spec.qos = Some(name.into());
            super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap();
            assert_eq!(spec.qos.as_deref(), Some(name));
        }
    }

    #[test]
    fn apply_default_qos_explicit_rejects_qos_outside_the_allow_list() {
        let assoc = AssociationCache::new();
        assoc.insert_allowed_qos("testuser", "research", &["a", "b"]);
        let qos = qos_cache_with(&["a", "b", "other-teams-qos"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());
        spec.qos = Some("other-teams-qos".into());

        let err = super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap_err();
        assert_eq!(
            err,
            SubmitError::invalid(
                "QOS 'other-teams-qos' is not permitted for user 'testuser' under account 'research'"
            )
        );
    }

    #[test]
    fn apply_default_qos_no_explicit_qos_resolves_pinned_default_from_allow_list() {
        // The common real-world config: an allow-list plus a default that's
        // a member of it. Omitting --qos must still resolve to the default.
        let assoc = AssociationCache::new();
        assoc.insert_allowed_qos("testuser", "research", &["a", "b"]);
        assoc.insert_default_qos("testuser", "research", "b");
        let qos = qos_cache_with(&["a", "b"]);
        let mut spec = basic_spec("j");
        spec.account = Some("research".into());

        super::apply_default_qos(&mut spec, &assoc, &qos, &acct_cfg()).unwrap();
        assert_eq!(spec.qos.as_deref(), Some("b"));
    }

    // ── array-parent dependency: cancel + display synthesis ──────

    /// Submit an array task job directly via the WAL (bypassing expansion) so
    /// tests can construct specific parent/task topologies.
    fn submit_array_task(cm: &ClusterManager, id: JobId, parent: JobId, task: u32) {
        let mut spec = basic_spec("arr");
        spec.array_job_id = Some(parent);
        spec.array_task_id = Some(task);
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: id,
            spec: Box::new(spec),
        });
    }

    fn set_terminal(cm: &ClusterManager, id: JobId, state: JobState, exit_code: i32) {
        // Jobs may only reach Completed/Failed/etc. via Running; cancel is the
        // only legal direct transition out of Pending.
        if state != JobState::Cancelled {
            cm.apply_operation(&WalOperation::job_state_change(
                id,
                JobState::Pending,
                JobState::Running,
            ));
        }
        cm.apply_operation(&WalOperation::JobComplete {
            job_id: id,
            exit_code,
            state,
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_unsatisfiable_dep_cancels_failed_afterok() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // Parent scalar job that fails.
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("parent")),
        });
        set_terminal(&cm, 1, JobState::Failed, 1);

        // Child depends on afterok:1 — can never be satisfied.
        let mut child = basic_spec("child");
        child.dependency = vec!["afterok:1".into()];
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 2,
            spec: Box::new(child),
        });

        let cancelled = cm.cancel_unsatisfiable_dependency_jobs();
        assert_eq!(cancelled, vec![2]);
        assert_eq!(cm.get_job(2).unwrap().state, JobState::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_unsatisfiable_dep_skips_running_job() {
        // A Running job with an unsatisfiable dep must not be cancelled
        // (Running -> Cancelled would destroy live work).
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("parent")),
        });
        set_terminal(&cm, 1, JobState::Failed, 1);

        let mut child = basic_spec("child");
        child.dependency = vec!["afterok:1".into()];
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 2,
            spec: Box::new(child),
        });
        // Child is already Running by the time the cancel pass fires.
        cm.apply_operation(&WalOperation::job_state_change(
            2,
            JobState::Pending,
            JobState::Running,
        ));

        let cancelled = cm.cancel_unsatisfiable_dependency_jobs();
        assert!(cancelled.is_empty(), "running job must not be cancelled");
        assert_eq!(cm.get_job(2).unwrap().state, JobState::Running);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_unsatisfiable_dep_tags_waiting_jobs() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // Parent still running; child waits, not cancelled.
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("parent")),
        });
        cm.apply_operation(&WalOperation::job_state_change(
            1,
            JobState::Pending,
            JobState::Running,
        ));

        let mut child = basic_spec("child");
        child.dependency = vec!["afterok:1".into()];
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 2,
            spec: Box::new(child),
        });

        let cancelled = cm.cancel_unsatisfiable_dependency_jobs();
        assert!(cancelled.is_empty());
        let child = cm.get_job(2).unwrap();
        assert_eq!(child.state, JobState::Pending);
        assert_eq!(child.pending_reason, PendingReason::Dependency);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_unsatisfiable_dep_array_parent_all_completed_releases() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // Array parent id 10, tasks 11/12/13 all completed.
        submit_array_task(&cm, 11, 10, 0);
        submit_array_task(&cm, 12, 10, 1);
        submit_array_task(&cm, 13, 10, 2);
        for id in [11, 12, 13] {
            set_terminal(&cm, id, JobState::Completed, 0);
        }

        // Child depends on afterok:10 (the array parent) — should be satisfied,
        // so neither cancelled nor tagged.
        let mut child = basic_spec("child");
        child.dependency = vec!["afterok:10".into()];
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 20,
            spec: Box::new(child),
        });

        let cancelled = cm.cancel_unsatisfiable_dependency_jobs();
        assert!(cancelled.is_empty());
        let child = cm.get_job(20).unwrap();
        assert_eq!(child.state, JobState::Pending);
        assert_ne!(child.pending_reason, PendingReason::Dependency);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_unsatisfiable_dep_array_parent_one_failed_cancels() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        submit_array_task(&cm, 11, 10, 0);
        submit_array_task(&cm, 12, 10, 1);
        set_terminal(&cm, 11, JobState::Completed, 0);
        set_terminal(&cm, 12, JobState::Failed, 1);

        let mut child = basic_spec("child");
        child.dependency = vec!["afterok:10".into()];
        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 20,
            spec: Box::new(child),
        });

        let cancelled = cm.cancel_unsatisfiable_dependency_jobs();
        assert_eq!(cancelled, vec![20]);
        assert_eq!(cm.get_job(20).unwrap().state, JobState::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_job_for_display_synthesizes_array_parent() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // No stored job with id 10; tasks 11/12 carry array_job_id=10.
        submit_array_task(&cm, 11, 10, 0);
        submit_array_task(&cm, 12, 10, 1);

        // Unfinished → aggregate Pending, no exit_code.
        let synth = cm
            .get_job_for_display(10)
            .expect("array parent should synthesize");
        assert_eq!(synth.job_id, 10);
        assert_eq!(synth.state, JobState::Pending);
        assert_eq!(synth.spec.array_job_id, Some(10));
        assert_eq!(synth.spec.array_task_id, None);
        assert_eq!(synth.exit_code, None);

        // Complete both → aggregate Completed, exit_code 0.
        set_terminal(&cm, 11, JobState::Completed, 0);
        set_terminal(&cm, 12, JobState::Completed, 0);
        let synth = cm.get_job_for_display(10).unwrap();
        assert_eq!(synth.state, JobState::Completed);
        assert_eq!(synth.exit_code, Some(0));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_job_for_display_scalar_and_unknown() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.apply_operation(&WalOperation::JobSubmit {
            job_id: 1,
            spec: Box::new(basic_spec("scalar")),
        });
        // Stored scalar job returned as-is.
        assert_eq!(cm.get_job_for_display(1).unwrap().job_id, 1);
        // Unknown id → None.
        assert!(cm.get_job_for_display(999).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_jobs_by_id_synthesizes_array_parent() {
        // `scontrol show job <parent>` / squeue go through the get_jobs list
        // RPC, not get_job. A query for the array parent id must return the
        // synthesized aggregate, not an empty list.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        submit_array_task(&cm, 11, 10, 0);
        submit_array_task(&cm, 12, 10, 1);

        // Query the parent id explicitly.
        let got = cm.get_jobs(&JobFilter {
            job_ids: &[10],
            ..Default::default()
        });
        assert_eq!(got.len(), 1, "parent id should synthesize one record");
        assert_eq!(got[0].job_id, 10);
        assert_eq!(got[0].state, JobState::Pending);
        assert_eq!(got[0].spec.array_job_id, Some(10));

        // Querying a real task id still returns that task, not the parent.
        let got_task = cm.get_jobs(&JobFilter {
            job_ids: &[11],
            ..Default::default()
        });
        assert_eq!(got_task.len(), 1);
        assert_eq!(got_task[0].job_id, 11);

        // Unknown id → empty.
        assert!(cm
            .get_jobs(&JobFilter {
                job_ids: &[999],
                ..Default::default()
            })
            .is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_jobs_filters_by_name() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        submit_and_wait(&cm, basic_spec("alpha"));
        submit_and_wait(&cm, basic_spec("beta"));
        submit_and_wait(&cm, basic_spec("alpha"));

        let by_name = |name: Option<&str>| {
            cm.get_jobs(&JobFilter {
                name,
                ..Default::default()
            })
        };

        let all = by_name(None);
        assert_eq!(all.len(), 3);

        let alphas = by_name(Some("alpha"));
        assert_eq!(alphas.len(), 2);
        assert!(alphas.iter().all(|j| j.spec.name == "alpha"));

        let betas = by_name(Some("beta"));
        assert_eq!(betas.len(), 1);

        let multi = by_name(Some("alpha,beta"));
        assert_eq!(multi.len(), 3);

        let none = by_name(Some("nonexistent"));
        assert!(none.is_empty());

        let empty = by_name(Some(""));
        assert_eq!(empty.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_jobs_filters_by_node() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);
        register_node(&cm, "n2", 8, 16000);
        let res = scalar_alloc(2, 4000);

        let id1 = submit_and_wait(&cm, basic_spec("on-n1"));
        cm.start_job(
            id1,
            vec!["n1".into()],
            res.clone(),
            per_node_for(&["n1"], res.clone()),
        )
        .unwrap();
        settle(&cm, id1, JobState::Running);

        let id2 = submit_and_wait(&cm, basic_spec("on-n2"));
        cm.start_job(
            id2,
            vec!["n2".into()],
            res.clone(),
            per_node_for(&["n2"], res.clone()),
        )
        .unwrap();
        settle(&cm, id2, JobState::Running);

        // Pending job has no allocated nodes and must never match a node filter.
        let id3 = submit_and_wait(&cm, basic_spec("pending"));

        let by_nodes = |nodes: &[String]| {
            cm.get_jobs(&JobFilter {
                nodes,
                ..Default::default()
            })
        };

        // Empty node filter is a no-op.
        assert_eq!(cm.get_jobs(&JobFilter::default()).len(), 3);

        let on_n1 = by_nodes(&["n1".to_string()]);
        assert_eq!(on_n1.len(), 1);
        assert_eq!(on_n1[0].job_id, id1);

        // Filtering by n2 excludes the running n1 job and the pending job.
        let on_n2 = by_nodes(&["n2".to_string()]);
        assert_eq!(on_n2.len(), 1);
        assert_eq!(on_n2[0].job_id, id2);

        // Union of both nodes returns both running jobs, never the pending one.
        let both = by_nodes(&["n1".to_string(), "n2".to_string()]);
        assert_eq!(both.len(), 2);
        assert!(!both.iter().any(|j| j.job_id == id3));

        // Unknown node matches nothing.
        assert!(by_nodes(&["n3".to_string()]).is_empty());
    }

    // --- Partition matching tests ---

    #[test]
    fn partition_matches_node_by_hostlist() {
        let part = Partition {
            name: "gpu".into(),
            nodes: "node[1-3]".into(),
            ..Default::default()
        };
        let empty_labels = HashMap::new();
        assert!(super::partition_matches_node(&part, "node1", &empty_labels));
        assert!(super::partition_matches_node(&part, "node3", &empty_labels));
        assert!(!super::partition_matches_node(
            &part,
            "node4",
            &empty_labels
        ));
    }

    #[test]
    fn partition_matches_node_by_selector() {
        let mut selector = HashMap::new();
        selector.insert("pool".into(), "train".into());
        let part = Partition {
            name: "train".into(),
            selector,
            ..Default::default()
        };
        let mut labels = HashMap::new();
        labels.insert("pool".into(), "train".into());
        labels.insert("gpu".into(), "mi300x".into());
        assert!(super::partition_matches_node(
            &part,
            "arbitrary-host",
            &labels
        ));

        let wrong_labels = HashMap::from([("pool".into(), "infer".into())]);
        assert!(!super::partition_matches_node(
            &part,
            "arbitrary-host",
            &wrong_labels
        ));
    }

    #[test]
    fn partition_matches_node_union_of_both() {
        let mut selector = HashMap::new();
        selector.insert("pool".into(), "train".into());
        let part = Partition {
            name: "gpu".into(),
            nodes: "node1".into(),
            selector,
            ..Default::default()
        };
        // Matches by hostlist alone
        assert!(super::partition_matches_node(
            &part,
            "node1",
            &HashMap::new()
        ));
        // Matches by selector alone
        let labels = HashMap::from([("pool".into(), "train".into())]);
        assert!(super::partition_matches_node(&part, "other-host", &labels));
        // Matches neither
        assert!(!super::partition_matches_node(
            &part,
            "other-host",
            &HashMap::new()
        ));
    }

    #[test]
    fn node_config_matches_by_selector() {
        let nc = spur_core::config::NodeConfig {
            names: String::new(),
            selector: HashMap::from([("gpu".into(), "mi300x".into())]),
            cpus: 0,
            memory_mb: 0,
            gres: Vec::new(),
            features: Vec::new(),
            address: None,
            weight: 1,
        };
        let labels = HashMap::from([("gpu".into(), "mi300x".into())]);
        assert!(super::node_config_matches(&nc, "any-host", &labels));
        assert!(!super::node_config_matches(
            &nc,
            "any-host",
            &HashMap::new()
        ));
    }

    // --- Label update + partition re-routing ---

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_labels_reroutes_partition() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions = vec![
            spur_core::config::PartitionConfig {
                name: "default".into(),
                default: true,
                state: "UP".into(),
                nodes: "ALL".into(),
                selector: HashMap::new(),
                max_time: None,
                default_time: None,
                max_nodes: None,
                min_nodes: 1,
                allow_accounts: Vec::new(),
                allow_groups: Vec::new(),
                deny_accounts: Vec::new(),
                deny_qos: Vec::new(),
                allow_qos: Vec::new(),
                priority_tier: 1,
                preempt_mode: String::new(),
                preempt_exempt_time: None,
            },
            spur_core::config::PartitionConfig {
                name: "train".into(),
                default: false,
                state: "UP".into(),
                nodes: String::new(),
                selector: HashMap::from([("pool".into(), "train".into())]),
                max_time: None,
                default_time: None,
                max_nodes: None,
                min_nodes: 1,
                allow_accounts: Vec::new(),
                allow_groups: Vec::new(),
                deny_accounts: Vec::new(),
                deny_qos: Vec::new(),
                allow_qos: Vec::new(),
                priority_tier: 1,
                preempt_mode: String::new(),
                preempt_exempt_time: None,
            },
        ];
        let cm = Arc::new(ClusterManager::new(cfg, dir.path()).unwrap());
        let handle = crate::raft::start_raft(1, &["[::1]:0".into()], dir.path(), cm.clone())
            .await
            .unwrap();
        handle
            .raft
            .wait(Some(std::time::Duration::from_secs(5)))
            .metrics(|m| m.current_leader == Some(1), "leader elected")
            .await
            .unwrap();
        cm.set_raft(handle.raft);

        register_node(&cm, "worker1", 4, 8000);
        let node = cm.get_node("worker1").unwrap();
        // Initially only in "default" (ALL matches everything)
        assert!(node.partitions.contains(&"default".into()));
        assert!(!node.partitions.contains(&"train".into()));

        // Add label that matches "train" partition selector
        cm.update_node_labels(
            "worker1",
            HashMap::from([("pool".into(), "train".into())]),
            &[],
        )
        .unwrap();
        wait_for("label applied", || {
            cm.get_node("worker1")
                .map(|n| !n.labels.is_empty())
                .unwrap_or(false)
        });

        let node = cm.get_node("worker1").unwrap();
        assert!(node.partitions.contains(&"train".into()));
        assert_eq!(node.labels.get("pool"), Some(&"train".into()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn register_node_with_labels_gets_selector_partition() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.partitions = vec![spur_core::config::PartitionConfig {
            name: "inference".into(),
            default: false,
            state: "UP".into(),
            nodes: String::new(),
            selector: HashMap::from([("role".into(), "infer".into())]),
            max_time: None,
            default_time: None,
            max_nodes: None,
            min_nodes: 1,
            allow_accounts: Vec::new(),
            allow_groups: Vec::new(),
            deny_accounts: Vec::new(),
            deny_qos: Vec::new(),
            allow_qos: Vec::new(),
            priority_tier: 1,
            preempt_mode: String::new(),
            preempt_exempt_time: None,
        }];
        let cm = Arc::new(ClusterManager::new(cfg, dir.path()).unwrap());
        let handle = crate::raft::start_raft(1, &["[::1]:0".into()], dir.path(), cm.clone())
            .await
            .unwrap();
        handle
            .raft
            .wait(Some(std::time::Duration::from_secs(5)))
            .metrics(|m| m.current_leader == Some(1), "leader elected")
            .await
            .unwrap();
        cm.set_raft(handle.raft);

        cm.register_node(
            "dyn-node".into(),
            "dyn-node".into(),
            ResourceSet {
                cpus: 8,
                memory_mb: 16000,
                ..Default::default()
            },
            "127.0.0.1".into(),
            6818,
            String::new(),
            String::new(),
            spur_core::node::NodeSource::NativeHost,
            HashMap::from([("role".into(), "infer".into())]),
            true,
        )
        .unwrap();
        wait_for("node registered", || cm.get_node("dyn-node").is_some());

        let node = cm.get_node("dyn-node").unwrap();
        assert!(node.partitions.contains(&"inference".into()));
    }

    #[test]
    fn partition_all_matches_any_node() {
        let part = Partition {
            name: "everything".into(),
            nodes: "ALL".into(),
            ..Default::default()
        };
        assert!(super::partition_matches_node(
            &part,
            "random-host-xyz",
            &HashMap::new()
        ));
        assert!(super::partition_matches_node(
            &part,
            "node1",
            &HashMap::new()
        ));
    }

    #[test]
    fn node_config_all_matches_any_node() {
        let nc = spur_core::config::NodeConfig {
            names: "ALL".into(),
            selector: HashMap::new(),
            cpus: 0,
            memory_mb: 0,
            gres: Vec::new(),
            features: vec!["common".into()],
            address: None,
            weight: 1,
        };
        assert!(super::node_config_matches(&nc, "any-host", &HashMap::new()));
        assert!(super::node_config_matches(
            &nc,
            "another",
            &HashMap::from([("x".into(), "y".into())])
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reregistration_syncs_labels() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // First registration with labels
        cm.register_node(
            "worker1".into(),
            "worker1".into(),
            ResourceSet {
                cpus: 4,
                memory_mb: 8000,
                ..Default::default()
            },
            "127.0.0.1".into(),
            6818,
            String::new(),
            String::new(),
            spur_core::node::NodeSource::NativeHost,
            HashMap::from([("pool".into(), "train".into())]),
            true,
        )
        .unwrap();
        wait_for("node registered", || cm.get_node("worker1").is_some());
        assert_eq!(
            cm.get_node("worker1").unwrap().labels.get("pool"),
            Some(&"train".into())
        );

        // Re-register with same resources but different labels
        cm.register_node(
            "worker1".into(),
            "worker1".into(),
            ResourceSet {
                cpus: 4,
                memory_mb: 8000,
                ..Default::default()
            },
            "127.0.0.1".into(),
            6818,
            String::new(),
            String::new(),
            spur_core::node::NodeSource::NativeHost,
            HashMap::from([("pool".into(), "infer".into()), ("tier".into(), "1".into())]),
            true,
        )
        .unwrap();
        wait_for("labels synced", || {
            cm.get_node("worker1")
                .map(|n| n.labels.get("pool") == Some(&"infer".into()))
                .unwrap_or(false)
        });

        let node = cm.get_node("worker1").unwrap();
        assert_eq!(node.labels.get("pool"), Some(&"infer".into()));
        assert_eq!(node.labels.get("tier"), Some(&"1".into()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reregistration_updates_comm_address_without_resource_change() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let resources = ResourceSet {
            cpus: 4,
            memory_mb: 8000,
            ..Default::default()
        };

        cm.register_node(
            "worker1".into(),
            "worker1".into(),
            resources.clone(),
            "10.0.0.1".into(),
            6818,
            String::new(),
            String::new(),
            spur_core::node::NodeSource::NativeHost,
            HashMap::new(),
            true,
        )
        .unwrap();
        wait_for("node registered", || cm.get_node("worker1").is_some());

        cm.register_node(
            "worker1".into(),
            "worker1".into(),
            resources,
            "10.0.0.2".into(),
            6818,
            String::new(),
            String::new(),
            spur_core::node::NodeSource::NativeHost,
            HashMap::new(),
            true,
        )
        .unwrap();
        wait_for("comm address updated", || {
            cm.get_node("worker1").and_then(|n| n.address).as_deref() == Some("10.0.0.2")
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_config_address_does_not_override_registered_comm_addr() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.nodes = vec![spur_core::config::NodeConfig {
            names: "worker1".into(),
            selector: HashMap::new(),
            cpus: 0,
            memory_mb: 0,
            gres: Vec::new(),
            features: Vec::new(),
            address: Some("10.0.0.99".into()),
            weight: 1,
        }];
        let cm = test_cluster_with_config(&dir, cfg).await;

        cm.register_node(
            "worker1".into(),
            "worker1".into(),
            ResourceSet {
                cpus: 4,
                memory_mb: 8000,
                ..Default::default()
            },
            "10.0.0.1".into(),
            6818,
            String::new(),
            String::new(),
            spur_core::node::NodeSource::NativeHost,
            HashMap::from([("pool".into(), "train".into())]),
            true,
        )
        .unwrap();
        wait_for("node registered", || cm.get_node("worker1").is_some());

        cm.apply_operation(&WalOperation::NodeLabelsUpdate {
            name: "worker1".into(),
            set: HashMap::from([("pool".into(), "infer".into())]),
            remove: Vec::new(),
        });

        assert_eq!(
            cm.get_node("worker1").unwrap().address,
            Some("10.0.0.1".into())
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_config_address_fills_when_agent_address_empty() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.nodes = vec![spur_core::config::NodeConfig {
            names: "worker1".into(),
            selector: HashMap::new(),
            cpus: 0,
            memory_mb: 0,
            gres: Vec::new(),
            features: Vec::new(),
            address: Some("10.0.0.99".into()),
            weight: 1,
        }];
        let cm = test_cluster_with_config(&dir, cfg).await;

        cm.apply_operation(&WalOperation::NodeRegister {
            name: "worker1".into(),
            hostname: "worker1".into(),
            resources: ResourceSet {
                cpus: 4,
                memory_mb: 8000,
                ..Default::default()
            },
            address: String::new(),
            port: 6818,
            wg_pubkey: String::new(),
            version: String::new(),
            labels: HashMap::new(),
            source: spur_core::node::NodeSource::NativeHost,
        });

        assert_eq!(
            cm.get_node("worker1").unwrap().address,
            Some("10.0.0.99".into())
        );
    }

    #[test]
    fn label_update_applies_nodeconfig_features() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.nodes = vec![spur_core::config::NodeConfig {
            names: String::new(),
            selector: HashMap::from([("gpu".into(), "mi300x".into())]),
            cpus: 0,
            memory_mb: 0,
            gres: Vec::new(),
            features: vec!["mi300x".into(), "rocm6".into()],
            address: None,
            weight: 10,
        }];
        let cm = ClusterManager::new(cfg, dir.path()).unwrap();

        // Register a node directly via WAL apply
        cm.apply_operation(&WalOperation::NodeRegister {
            name: "gpu-node".into(),
            hostname: String::new(),
            resources: ResourceSet {
                cpus: 8,
                memory_mb: 16000,
                ..Default::default()
            },
            address: "127.0.0.1".into(),
            port: 6818,
            wg_pubkey: String::new(),
            version: String::new(),
            labels: HashMap::new(),
            source: NodeSource::default(),
        });

        let node = cm.get_node("gpu-node").unwrap();
        assert!(node.features.is_empty());

        // Apply label update that matches the NodeConfig selector
        cm.apply_operation(&WalOperation::NodeLabelsUpdate {
            name: "gpu-node".into(),
            set: HashMap::from([("gpu".into(), "mi300x".into())]),
            remove: Vec::new(),
        });

        let node = cm.get_node("gpu-node").unwrap();
        assert_eq!(node.features, vec!["mi300x", "rocm6"]);
        assert_eq!(node.weight, 10);
    }

    #[test]
    fn label_update_resets_features_when_no_match() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.nodes = vec![spur_core::config::NodeConfig {
            names: String::new(),
            selector: HashMap::from([("gpu".into(), "mi300x".into())]),
            cpus: 0,
            memory_mb: 0,
            gres: Vec::new(),
            features: vec!["mi300x".into(), "rocm6".into()],
            address: None,
            weight: 10,
        }];
        let cm = ClusterManager::new(cfg, dir.path()).unwrap();

        cm.apply_operation(&WalOperation::NodeRegister {
            name: "gpu-node".into(),
            hostname: String::new(),
            resources: ResourceSet {
                cpus: 8,
                memory_mb: 16000,
                ..Default::default()
            },
            address: "127.0.0.1".into(),
            port: 6818,
            wg_pubkey: String::new(),
            version: String::new(),
            labels: HashMap::from([("gpu".into(), "mi300x".into())]),
            source: NodeSource::default(),
        });

        let node = cm.get_node("gpu-node").unwrap();
        assert_eq!(node.features, vec!["mi300x", "rocm6"]);
        assert_eq!(node.weight, 10);

        cm.apply_operation(&WalOperation::NodeLabelsUpdate {
            name: "gpu-node".into(),
            set: HashMap::new(),
            remove: vec!["gpu".into()],
        });

        let node = cm.get_node("gpu-node").unwrap();
        assert!(node.features.is_empty());
        assert_eq!(node.weight, 1);
    }

    #[test]
    fn node_register_no_match_uses_defaults() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.nodes = vec![spur_core::config::NodeConfig {
            names: String::new(),
            selector: HashMap::from([("gpu".into(), "mi300x".into())]),
            cpus: 0,
            memory_mb: 0,
            gres: Vec::new(),
            features: vec!["mi300x".into(), "rocm6".into()],
            address: None,
            weight: 10,
        }];
        let cm = ClusterManager::new(cfg, dir.path()).unwrap();

        cm.apply_operation(&WalOperation::NodeRegister {
            name: "cpu-node".into(),
            hostname: String::new(),
            resources: ResourceSet {
                cpus: 8,
                memory_mb: 16000,
                ..Default::default()
            },
            address: "127.0.0.1".into(),
            port: 6818,
            wg_pubkey: String::new(),
            version: String::new(),
            labels: HashMap::from([("gpu".into(), "mi250".into())]),
            source: NodeSource::default(),
        });

        let node = cm.get_node("cpu-node").unwrap();
        assert!(node.features.is_empty());
        assert_eq!(node.weight, 1);
    }

    #[test]
    fn reconcile_resets_stale_features_on_restore() {
        let dir = TempDir::new().unwrap();
        let mut cfg = test_config();
        cfg.nodes = vec![spur_core::config::NodeConfig {
            names: String::new(),
            selector: HashMap::from([("gpu".into(), "mi300x".into())]),
            cpus: 0,
            memory_mb: 0,
            gres: Vec::new(),
            features: vec!["mi300x".into(), "rocm6".into()],
            address: None,
            weight: 10,
        }];
        let cm = ClusterManager::new(cfg, dir.path()).unwrap();

        // Snapshot node has stale policy but labels that no longer match the config.
        let mut stale = Node::new(
            "gpu-node".into(),
            ResourceSet {
                cpus: 8,
                memory_mb: 16000,
                ..Default::default()
            },
        );
        stale.features = vec!["mi300x".into(), "rocm6".into()];
        stale.weight = 10;
        stale.labels = HashMap::new();

        let snap = ClusterSnapshot {
            jobs: Vec::new(),
            nodes: vec![stale],
            reservations: Vec::new(),
            partitions: None,
            deleted_partition_names: HashSet::new(),
            steps: Vec::new(),
            license_pool: HashMap::new(),
            tokens: Vec::new(),
            burst_buffer_total_gb: 0,
            k0s: spur_core::k0s::K0sClusterState::default(),
            next_job_id: 0,
        };
        let bytes = serde_json::to_vec(&snap).unwrap();
        cm.restore_from_snapshot(&bytes).unwrap();

        let node = cm.get_node("gpu-node").unwrap();
        assert!(node.features.is_empty());
        assert_eq!(node.weight, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_triggers_scheduler_notify() {
        // Verify that submit_job() actually calls notify_one() in production code path.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // Set up a listener before submitting
        let notify = cm.scheduler_notify.clone();
        let listener = tokio::spawn(async move {
            notify.notified().await;
        });

        // Give listener time to register
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;

        // Submit a job - this should trigger notify_one()
        let spec = basic_spec("test");
        let _ = submit_and_wait(&cm, spec);

        // Verify notification was received (with timeout to prevent hanging)
        let result = tokio::time::timeout(tokio::time::Duration::from_millis(100), listener).await;

        assert!(
            result.is_ok(),
            "submit_job should call notify_one() to wake scheduler"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_job_notifies_even_with_array_expansion() {
        // Array jobs expand into multiple tasks; verify notify is called during expansion.
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // Set up a listener before submitting
        let notify = cm.scheduler_notify.clone();
        let listener = tokio::spawn(async move {
            notify.notified().await;
        });

        // Submit an array job (expands to multiple tasks). `submit_job` returns the
        // array parent id, which is not stored — only per-task ids exist in `jobs`.
        let mut spec = basic_spec("array");
        spec.array_spec = Some("0-2".into()); // Creates 3 tasks
        let parent_id = cm.submit_job(spec).unwrap().job_id;
        let first_task_id = parent_id + 1;
        wait_for(&format!("array task {first_task_id} applied"), || {
            cm.get_job(first_task_id).is_some()
        });

        // Verify notification was received (with timeout to prevent hanging)
        let result = tokio::time::timeout(tokio::time::Duration::from_secs(1), listener).await;
        assert!(
            result.is_ok(),
            "array job submission should trigger scheduler notification"
        );
    }

    // ---- Node deregistration tests ----

    fn start_job_on(cm: &ClusterManager, id: JobId, node: &str) {
        cm.apply_operation(&WalOperation::job_state_change(
            id,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobStart {
            job_id: id,
            nodes: vec![node.into()],
            resources: scalar_alloc(1, 1000),
            per_node_alloc: per_node_for(&[node], scalar_alloc(1, 1000)),
            srun_step_dispatch: false,
            run_attempt: 0,
        });
    }

    fn start_srun_job_on(cm: &ClusterManager, id: JobId, node: &str) {
        cm.apply_operation(&WalOperation::job_state_change(
            id,
            JobState::Pending,
            JobState::Running,
        ));
        cm.apply_operation(&WalOperation::JobStart {
            job_id: id,
            nodes: vec![node.into()],
            resources: scalar_alloc(1, 1000),
            per_node_alloc: per_node_for(&[node], scalar_alloc(1, 1000)),
            srun_step_dispatch: true,
            run_attempt: 0,
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_state_change_to_down_evicts_running_jobs() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("evict-me"));
        start_job_on(&cm, id, "n1");
        assert_eq!(cm.get_job(id).unwrap().state, JobState::Running);

        let resp = cm.apply_operation(&WalOperation::NodeStateChange {
            name: "n1".into(),
            old_state: NodeState::Allocated,
            new_state: NodeState::Down,
            reason: Some("heartbeat timeout".into()),
            admin_locked: false,
            reason_uid: None,
            reason_time: None,
        });
        assert_eq!(resp.jobs_finalized.len(), 1);
        assert_eq!(resp.jobs_finalized[0].job_id, id);
        assert_eq!(resp.jobs_finalized[0].state, JobState::NodeFail);

        let job = cm.get_job(id).unwrap();
        assert_eq!(job.state, JobState::NodeFail);
        assert_eq!(job.exit_code, Some(-1));

        let node = cm.get_node("n1").unwrap();
        assert_eq!(node.state, NodeState::Down);
        assert_eq!(node.alloc_resources.cpus, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_state_change_to_down_no_jobs_is_clean() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);

        let resp = cm.apply_operation(&WalOperation::NodeStateChange {
            name: "n1".into(),
            old_state: NodeState::Idle,
            new_state: NodeState::Down,
            reason: None,
            admin_locked: false,
            reason_uid: None,
            reason_time: None,
        });
        assert!(resp.jobs_finalized.is_empty());
        assert_eq!(cm.get_node("n1").unwrap().state, NodeState::Down);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_remove_deletes_node() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);
        assert!(cm.get_node("n1").is_some());

        cm.apply_operation(&WalOperation::NodeRemove {
            name: "n1".into(),
            reason: Some("decommission".into()),
        });
        assert!(cm.get_node("n1").is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn node_remove_evicts_and_deletes() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("j"));
        start_job_on(&cm, id, "n1");

        let resp = cm.apply_operation(&WalOperation::NodeRemove {
            name: "n1".into(),
            reason: None,
        });
        assert_eq!(resp.jobs_finalized.len(), 1);
        assert_eq!(resp.jobs_finalized[0].state, JobState::NodeFail);
        assert_eq!(cm.get_job(id).unwrap().state, JobState::NodeFail);
        assert!(cm.get_node("n1").is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_node_sets_draining_with_running_jobs() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("drain-job"));
        start_job_on(&cm, id, "n1");

        cm.drain_node("n1", Some("maintenance".into()), None)
            .unwrap();
        wait_for("n1 draining", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Draining)
        });

        let node = cm.get_node("n1").unwrap();
        assert!(node.admin_locked);
        assert_eq!(node.state_reason.as_deref(), Some("maintenance"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_node_sets_drain_without_running_jobs() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);

        cm.drain_node("n1", None, None).unwrap();
        wait_for("n1 drain", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Drain)
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remove_node_rejects_running_without_force() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("j"));
        start_job_on(&cm, id, "n1");

        let err = cm.remove_node("n1", false, None);
        assert!(err.is_err());
        assert!(cm.get_node("n1").is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remove_node_force_evicts_and_removes() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("j"));
        start_job_on(&cm, id, "n1");

        cm.remove_node("n1", true, Some("bad node".into())).unwrap();
        wait_for("n1 removed", || cm.get_node("n1").is_none());

        assert_eq!(cm.get_job(id).unwrap().state, JobState::NodeFail);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn multinode_eviction_frees_all_nodes() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);
        register_node(&cm, "n2", 4, 8000);

        let id = submit_and_wait(&cm, basic_spec("multi"));

        let alloc = scalar_alloc(2, 2000);
        let per_node = per_node_for(&["n1", "n2"], scalar_alloc(1, 1000));
        cm.start_job(id, vec!["n1".into(), "n2".into()], alloc, per_node)
            .unwrap();
        settle(&cm, id, JobState::Running);

        assert_eq!(cm.get_node("n1").unwrap().alloc_resources.cpus, 1);
        assert_eq!(cm.get_node("n2").unwrap().alloc_resources.cpus, 1);

        let evicted = cm
            .remove_node("n1", true, Some("evict test".into()))
            .unwrap();
        wait_for("n1 removed", || cm.get_node("n1").is_none());

        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].job_id, id);
        assert_eq!(cm.get_job(id).unwrap().state, JobState::NodeFail);

        let n2 = cm.get_node("n2").unwrap();
        assert_eq!(
            n2.alloc_resources.cpus, 0,
            "peer node n2 must have allocations freed"
        );
        assert_eq!(n2.alloc_resources.memory_mb, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn draining_to_drain_on_last_job_complete() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("drain-job"));
        start_job_on(&cm, id, "n1");

        cm.drain_node("n1", None, None).unwrap();
        wait_for("n1 draining", || {
            cm.get_node("n1")
                .is_some_and(|n| n.state == NodeState::Draining)
        });

        cm.apply_operation(&WalOperation::JobNodeComplete {
            job_id: id,
            node_name: "n1".into(),
            exit_code: 0,
            signal: 0,
        });

        let node = cm.get_node("n1").unwrap();
        assert_eq!(node.state, NodeState::Drain);
    }

    // --- Direct evict_job unit tests ---

    fn make_running_job(job_id: JobId, nodes: &[&str], cpus_per_node: u32) -> Job {
        let mut spec = basic_spec("evict-test");
        spec.cpus_per_task = cpus_per_node;
        let mut job = Job::new(job_id, spec);
        job.state = JobState::Running;
        job.start_time = Some(Utc::now());
        let node_list: Vec<String> = nodes.iter().map(|n| (*n).to_string()).collect();
        let total_cpus = cpus_per_node * nodes.len() as u32;
        job.allocated_nodes = node_list;
        job.allocated_resources = Some(ResourceAllocations::with_scalar(total_cpus, 0));
        job.per_node_alloc = nodes
            .iter()
            .map(|n| {
                (
                    (*n).to_string(),
                    ResourceAllocations::with_scalar(cpus_per_node, 0),
                )
            })
            .collect();
        job
    }

    fn make_test_node(name: &str, total_cpus: u32, alloc_cpus: u32) -> Node {
        let mut node = Node::new(
            name.into(),
            ResourceSet {
                cpus: total_cpus,
                ..Default::default()
            },
        );
        node.state = if alloc_cpus > 0 {
            NodeState::Allocated
        } else {
            NodeState::Idle
        };
        node.alloc_resources = ResourceAllocations::with_scalar(alloc_cpus, 0);
        node
    }

    #[test]
    fn evict_job_returns_none_for_missing_job() {
        let mut jobs = HashMap::new();
        let mut nodes = HashMap::new();
        let result = ClusterManager::evict_job_locked(
            999,
            &mut jobs,
            &mut nodes,
            Utc::now(),
            PendingReason::NodeDown,
        );
        assert!(result.is_none());
    }

    #[test]
    fn evict_job_transitions_running_to_nodefail() {
        let mut jobs = HashMap::new();
        let mut nodes = HashMap::new();
        jobs.insert(1, make_running_job(1, &["n1"], 2));
        nodes.insert("n1".into(), make_test_node("n1", 4, 2));

        let fin = ClusterManager::evict_job_locked(
            1,
            &mut jobs,
            &mut nodes,
            Utc::now(),
            PendingReason::NodeDown,
        )
        .unwrap();
        assert_eq!(fin.job_id, 1);
        assert_eq!(fin.state, JobState::NodeFail);
        assert_eq!(fin.exit_code, -1);

        let job = &jobs[&1];
        assert_eq!(job.state, JobState::NodeFail);
        assert_eq!(job.exit_code, Some(-1));
        assert!(job.end_time.is_some());
        assert_eq!(job.pending_reason, PendingReason::NodeDown);
    }

    #[test]
    fn evict_job_frees_allocations_on_all_nodes() {
        let mut jobs = HashMap::new();
        let mut nodes = HashMap::new();
        jobs.insert(1, make_running_job(1, &["n1", "n2"], 2));
        nodes.insert("n1".into(), make_test_node("n1", 4, 2));
        nodes.insert("n2".into(), make_test_node("n2", 4, 2));

        ClusterManager::evict_job_locked(
            1,
            &mut jobs,
            &mut nodes,
            Utc::now(),
            PendingReason::NodeDown,
        );

        assert_eq!(nodes["n1"].alloc_resources.cpus, 0);
        assert_eq!(nodes["n2"].alloc_resources.cpus, 0);
    }

    #[test]
    fn evict_job_returns_none_for_terminal_job() {
        let mut jobs = HashMap::new();
        let mut nodes = HashMap::new();
        let mut job = make_running_job(1, &["n1"], 2);
        job.state = JobState::Completed;
        jobs.insert(1, job);
        nodes.insert("n1".into(), make_test_node("n1", 4, 2));

        let result = ClusterManager::evict_job_locked(
            1,
            &mut jobs,
            &mut nodes,
            Utc::now(),
            PendingReason::NodeDown,
        );
        assert!(result.is_none());
    }

    #[test]
    fn evict_job_finalizes_suspended_time() {
        let mut jobs = HashMap::new();
        let mut nodes = HashMap::new();
        let suspended_at = Utc::now() - chrono::Duration::seconds(30);
        let mut job = make_running_job(1, &["n1"], 2);
        job.state = JobState::Suspended;
        job.suspended_at = Some(suspended_at);
        job.suspended_secs = 10;
        jobs.insert(1, job);
        nodes.insert("n1".into(), make_test_node("n1", 4, 2));

        ClusterManager::evict_job_locked(
            1,
            &mut jobs,
            &mut nodes,
            Utc::now(),
            PendingReason::NodeDown,
        );

        let job = &jobs[&1];
        assert!(job.suspended_at.is_none());
        assert!(job.suspended_secs >= 40, "should accumulate ~30s more");
    }

    #[test]
    fn evict_job_transitions_completing_to_nodefail() {
        let mut jobs = HashMap::new();
        let mut nodes = HashMap::new();
        let mut job = make_running_job(1, &["n1"], 2);
        job.state = JobState::Completing;
        jobs.insert(1, job);
        nodes.insert("n1".into(), make_test_node("n1", 4, 2));

        let fin = ClusterManager::evict_job_locked(
            1,
            &mut jobs,
            &mut nodes,
            Utc::now(),
            PendingReason::NodeDown,
        )
        .unwrap();
        assert_eq!(fin.state, JobState::NodeFail);
        assert_eq!(jobs[&1].state, JobState::NodeFail);
        assert_eq!(nodes["n1"].alloc_resources.cpus, 0);
    }

    #[test]
    fn evict_job_transitions_draining_node_to_drain() {
        let mut jobs = HashMap::new();
        let mut nodes = HashMap::new();
        jobs.insert(1, make_running_job(1, &["n1"], 2));
        let mut node = make_test_node("n1", 4, 2);
        node.state = NodeState::Draining;
        nodes.insert("n1".into(), node);

        ClusterManager::evict_job_locked(
            1,
            &mut jobs,
            &mut nodes,
            Utc::now(),
            PendingReason::NodeDown,
        );

        assert_eq!(nodes["n1"].state, NodeState::Drain);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn finish_srun_job_completes_running_allocation() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, srun_spec("srun-alloc"));
        start_srun_job_on(&cm, id, "n1");

        let returned = cm.finish_srun_job(id, 0, "testuser").unwrap();
        assert_eq!(returned.job_id, id);
        settle(&cm, id, JobState::Completed);
        assert_eq!(cm.get_job(id).unwrap().exit_code, Some(0));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn finish_srun_job_rejects_non_srun_job() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, basic_spec("batch"));
        start_job_on(&cm, id, "n1");

        assert!(matches!(
            cm.finish_srun_job(id, 0, "testuser"),
            Err(SrunCompleteError::NotSrunJob(j)) if j == id
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn finish_srun_job_rejects_terminal_job() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, srun_spec("srun-done"));
        start_srun_job_on(&cm, id, "n1");
        cm.finish_srun_job(id, 0, "testuser").unwrap();
        settle(&cm, id, JobState::Completed);

        assert!(matches!(
            cm.finish_srun_job(id, 0, "testuser"),
            Err(SrunCompleteError::AlreadyTerminal {
                job_id,
                state: JobState::Completed,
            }) if job_id == id
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn finish_srun_job_rejects_wrong_user() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, srun_spec("srun-owner"));
        start_srun_job_on(&cm, id, "n1");

        assert!(matches!(
            cm.finish_srun_job(id, 0, "otheruser"),
            Err(SrunCompleteError::NotOwner { job_id, user }) if job_id == id && user == "otheruser"
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn finish_srun_job_rejects_batch_fallback_srun() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 4, 8000);
        let id = submit_and_wait(&cm, srun_spec("srun-batch-fallback"));
        start_job_on(&cm, id, "n1");

        assert!(matches!(
            cm.finish_srun_job(id, 0, "testuser"),
            Err(SrunCompleteError::NotStepDispatch(j)) if j == id
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn finish_srun_job_not_found() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        assert!(matches!(
            cm.finish_srun_job(999, 0, "testuser"),
            Err(SrunCompleteError::NotFound(999))
        ));
    }

    // ---------------------------------------------------------------
    // Partition CRUD tests
    // ---------------------------------------------------------------

    fn gpu_partition() -> spur_core::partition::Partition {
        spur_core::partition::Partition {
            name: "gpu".into(),
            state: spur_core::partition::PartitionState::Up,
            is_default: false,
            nodes: "gpu[01-04]".into(),
            max_time_minutes: Some(1440),
            allow_accounts: vec!["ml-team".into()],
            priority_tier: 2,
            ..Default::default()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_partition_create_update_delete() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.apply_operation(&WalOperation::PartitionCreate {
            partition: gpu_partition(),
        });

        let parts = cm.get_partitions();
        assert!(
            parts.iter().any(|p| p.name == "gpu"),
            "gpu partition missing after create"
        );
        let gpu = parts.iter().find(|p| p.name == "gpu").unwrap();
        assert_eq!(gpu.max_time_minutes, Some(1440));
        assert_eq!(gpu.allow_accounts, vec!["ml-team"]);
        assert_eq!(gpu.priority_tier, 2);

        cm.apply_operation(&WalOperation::PartitionUpdate {
            name: "gpu".into(),
            nodes: None,
            selector: None,
            state: Some("DRAIN".into()),
            max_time_minutes: Some(Some(2880)),
            default_time_minutes: None,
            max_nodes: None,
            min_nodes: None,
            allow_accounts: Some(vec!["ml-team".into(), "infra".into()]),
            allow_groups: None,
            deny_accounts: None,
            deny_qos: None,
            allow_qos: None,
            priority_tier: None,
            preempt_mode: None,
            is_default: None,
            preempt_exempt_time: None,
        });

        let gpu = cm
            .get_partitions()
            .into_iter()
            .find(|p| p.name == "gpu")
            .unwrap();
        assert_eq!(gpu.state, spur_core::partition::PartitionState::Drain);
        assert_eq!(gpu.max_time_minutes, Some(2880));
        assert!(gpu.allow_accounts.contains(&"infra".into()));

        cm.apply_operation(&WalOperation::PartitionDelete { name: "gpu".into() });
        assert!(
            !cm.get_partitions().iter().any(|p| p.name == "gpu"),
            "gpu partition still present after delete"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_partition_create_idempotent() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.apply_operation(&WalOperation::PartitionCreate {
            partition: gpu_partition(),
        });
        cm.apply_operation(&WalOperation::PartitionCreate {
            partition: gpu_partition(),
        });

        let gpu_count = cm
            .get_partitions()
            .iter()
            .filter(|p| p.name == "gpu")
            .count();
        assert_eq!(
            gpu_count, 1,
            "duplicate PartitionCreate must not add a second entry"
        );
    }

    // A config-seeded partition must lose to a replayed WAL PartitionCreate of
    // the same name, or a runtime edit later codified into spur.conf reverts on
    // restart and two controllers with differing confs diverge.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_partition_create_overrides_config_seed() {
        let dir = TempDir::new().unwrap();
        // Default test config seeds a partition named "default".
        let cm = test_cluster(&dir).await;
        assert!(
            cm.config_seeded_partitions.read().contains("default"),
            "precondition: 'default' is config-seeded"
        );

        let runtime = spur_core::partition::Partition {
            name: "default".into(),
            state: spur_core::partition::PartitionState::Up,
            is_default: true,
            nodes: "node[01-09]".into(),
            max_time_minutes: Some(720),
            allow_accounts: vec!["runtime-team".into()],
            priority_tier: 7,
            ..Default::default()
        };
        cm.apply_operation(&WalOperation::PartitionCreate {
            partition: runtime.clone(),
        });

        let parts = cm.get_partitions();
        let def = parts.iter().find(|p| p.name == "default").unwrap();
        assert_eq!(def.nodes, "node[01-09]", "WAL value must overwrite seed");
        assert_eq!(def.max_time_minutes, Some(720));
        assert_eq!(def.priority_tier, 7);
        assert_eq!(
            parts.iter().filter(|p| p.name == "default").count(),
            1,
            "override must not add a second entry"
        );
        assert!(
            !cm.config_seeded_partitions.read().contains("default"),
            "seed marker must be cleared once the WAL has overridden it"
        );

        // The seed override is one-shot: a further duplicate is a genuine
        // create race and stays first-writer-wins.
        let mut evil = runtime.clone();
        evil.priority_tier = 99;
        cm.apply_operation(&WalOperation::PartitionCreate { partition: evil });
        let def = cm
            .get_partitions()
            .into_iter()
            .find(|p| p.name == "default")
            .unwrap();
        assert_eq!(
            def.priority_tier, 7,
            "post-override duplicate must be ignored (first-writer-wins)"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_partition_update_unknown_is_a_noop() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let before = cm.get_partitions().len();

        cm.apply_operation(&WalOperation::PartitionUpdate {
            name: "does-not-exist".into(),
            nodes: Some("n1".into()),
            selector: None,
            state: None,
            max_time_minutes: None,
            default_time_minutes: None,
            max_nodes: None,
            min_nodes: None,
            allow_accounts: None,
            allow_groups: None,
            deny_accounts: None,
            deny_qos: None,
            allow_qos: None,
            priority_tier: None,
            preempt_mode: None,
            is_default: None,
            preempt_exempt_time: None,
        });

        assert_eq!(
            cm.get_partitions().len(),
            before,
            "unknown partition update must not add an entry"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_partition_set_default_clears_others() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        // Baseline: the "default" partition created by test_config() is already is_default=true.
        cm.apply_operation(&WalOperation::PartitionCreate {
            partition: gpu_partition(),
        });
        assert!(
            !cm.get_partitions()
                .iter()
                .find(|p| p.name == "gpu")
                .unwrap()
                .is_default
        );

        // Make "gpu" the default.
        cm.apply_operation(&WalOperation::PartitionUpdate {
            name: "gpu".into(),
            nodes: None,
            selector: None,
            state: None,
            max_time_minutes: None,
            default_time_minutes: None,
            max_nodes: None,
            min_nodes: None,
            allow_accounts: None,
            allow_groups: None,
            deny_accounts: None,
            deny_qos: None,
            allow_qos: None,
            priority_tier: None,
            preempt_mode: None,
            is_default: Some(true),
            preempt_exempt_time: None,
        });

        let parts = cm.get_partitions();
        let defaults: Vec<&str> = parts
            .iter()
            .filter(|p| p.is_default)
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(
            defaults,
            vec!["gpu"],
            "exactly one partition must be default after promotion"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_partition_rejects_duplicate_name() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        cm.create_partition(gpu_partition()).unwrap();
        let err = cm.create_partition(gpu_partition()).unwrap_err();
        assert!(
            matches!(err, PartitionError::AlreadyExists(_)),
            "expected AlreadyExists, got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_partition_rejects_not_found() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let err = cm.delete_partition("no-such-partition").unwrap_err();
        assert!(
            matches!(err, PartitionError::NotFound(_)),
            "expected NotFound, got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_partition_rejects_when_running_job_uses_it() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        register_node(&cm, "n1", 8, 16000);

        // Use an open partition (no account restriction) so basic_spec's testuser can submit.
        let open_part = spur_core::partition::Partition {
            name: "gpu".into(),
            nodes: "ALL".into(),
            ..Default::default()
        };
        cm.create_partition(open_part).unwrap();
        wait_for("gpu partition created", || {
            cm.get_partitions().iter().any(|p| p.name == "gpu")
        });

        let mut spec = basic_spec("gpu-job");
        spec.partition = Some("gpu".into());
        let job_id = submit_and_wait(&cm, spec);
        let alloc = scalar_alloc(1, 1000);
        cm.start_job(
            job_id,
            vec!["n1".into()],
            alloc.clone(),
            per_node_for(&["n1"], alloc),
        )
        .unwrap();
        settle(&cm, job_id, JobState::Running);

        let err = cm.delete_partition("gpu").unwrap_err();
        assert!(
            matches!(err, PartitionError::InvalidArgument(_)),
            "expected InvalidArgument for in-use partition, got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn partition_survives_wal_replay() {
        let dir = TempDir::new().unwrap();
        {
            let cm = test_cluster(&dir).await;
            cm.create_partition(gpu_partition()).unwrap();
            wait_for("gpu created", || {
                cm.get_partitions().iter().any(|p| p.name == "gpu")
            });
        }

        let cm2 = test_cluster(&dir).await;
        wait_for("gpu replayed from WAL", || {
            cm2.get_partitions().iter().any(|p| p.name == "gpu")
        });
        let gpu = cm2
            .get_partitions()
            .into_iter()
            .find(|p| p.name == "gpu")
            .unwrap();
        assert_eq!(gpu.allow_accounts, vec!["ml-team"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_partition_create_keeps_single_entry() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        let cm1 = cm.clone();
        let cm2 = cm.clone();
        let (first, second) = tokio::join!(
            tokio::task::spawn_blocking(move || cm1.create_partition(gpu_partition())),
            tokio::task::spawn_blocking(move || cm2.create_partition(gpu_partition())),
        );
        let outcomes = [first.unwrap(), second.unwrap()];
        assert_eq!(
            outcomes.iter().filter(|r| r.is_ok()).count(),
            1,
            "exactly one concurrent create must succeed"
        );
        let gpu_count = cm
            .get_partitions()
            .iter()
            .filter(|p| p.name == "gpu")
            .count();
        assert_eq!(gpu_count, 1);
    }

    // ---------------------------------------------------------------
    // Tombstone tests
    // ---------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deleted_partition_does_not_resurface_after_snapshot_restore() {
        let state_dir = TempDir::new().unwrap();
        let cm = test_cluster(&state_dir).await;

        // Create then delete "gpu" via the public API (goes through Raft).
        cm.create_partition(gpu_partition()).unwrap();
        wait_for("gpu created", || {
            cm.get_partitions().iter().any(|p| p.name == "gpu")
        });
        cm.delete_partition("gpu").unwrap();
        wait_for("gpu deleted", || {
            !cm.get_partitions().iter().any(|p| p.name == "gpu")
        });

        // Tombstone set must contain "gpu".
        assert!(
            cm.deleted_partition_names.read().contains("gpu"),
            "tombstone not recorded after delete"
        );

        // Simulate snapshot + restore: take a snapshot, apply it to a fresh manager.
        let snap_bytes = cm.snapshot_state().unwrap();

        let cm2 = Arc::new(ClusterManager::new(test_config(), state_dir.path()).unwrap());
        cm2.restore_from_snapshot(&snap_bytes).unwrap();

        assert!(
            !cm2.get_partitions().iter().any(|p| p.name == "gpu"),
            "deleted partition must not re-appear after snapshot restore"
        );
        assert!(
            cm2.deleted_partition_names.read().contains("gpu"),
            "tombstone must survive snapshot round-trip"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recreating_tombstoned_partition_clears_tombstone() {
        let state_dir = TempDir::new().unwrap();
        let cm = test_cluster(&state_dir).await;

        cm.create_partition(gpu_partition()).unwrap();
        wait_for("gpu created", || {
            cm.get_partitions().iter().any(|p| p.name == "gpu")
        });

        cm.delete_partition("gpu").unwrap();
        wait_for("gpu deleted", || {
            !cm.get_partitions().iter().any(|p| p.name == "gpu")
        });
        assert!(cm.deleted_partition_names.read().contains("gpu"));

        // Recreate.
        cm.create_partition(gpu_partition()).unwrap();
        wait_for("gpu recreated", || {
            cm.get_partitions().iter().any(|p| p.name == "gpu")
        });

        assert!(
            !cm.deleted_partition_names.read().contains("gpu"),
            "tombstone must be cleared when the partition is recreated"
        );

        // After another snapshot round-trip the partition must be present.
        let snap_bytes = cm.snapshot_state().unwrap();
        let cm2 = Arc::new(ClusterManager::new(test_config(), state_dir.path()).unwrap());
        cm2.restore_from_snapshot(&snap_bytes).unwrap();
        assert!(
            cm2.get_partitions().iter().any(|p| p.name == "gpu"),
            "recreated partition must survive snapshot round-trip"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn config_partition_not_touched_by_runtime_stays_after_snapshot_restore() {
        // Verify that the existing deployment path is unaffected: a partition that
        // was only ever defined in spur.conf and never touched at runtime must
        // still be present after snapshot restore.
        let state_dir = TempDir::new().unwrap();
        let cm = test_cluster(&state_dir).await;

        // test_config() includes a "default" partition — never touch it at runtime.
        // Take a snapshot and restore.
        let snap_bytes = cm.snapshot_state().unwrap();
        let cm2 = Arc::new(ClusterManager::new(test_config(), state_dir.path()).unwrap());
        cm2.restore_from_snapshot(&snap_bytes).unwrap();

        assert!(
            cm2.get_partitions().iter().any(|p| p.name == "default"),
            "config-only partition must survive snapshot restore untouched"
        );
    }

    // ---------------------------------------------------------------
    // QoS enforcement tests
    // ---------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn validate_partition_rejects_qos_not_in_allow_qos() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let part = spur_core::partition::Partition {
            name: "restricted".into(),
            nodes: "ALL".into(),
            allow_qos: vec!["premium".into()],
            ..Default::default()
        };
        cm.create_partition(part).unwrap();
        wait_for("restricted created", || {
            cm.get_partitions().iter().any(|p| p.name == "restricted")
        });

        // QoS not in allow list → rejected.
        let mut spec = basic_spec("bad-qos");
        spec.partition = Some("restricted".into());
        spec.qos = Some("cheap".into());
        assert!(
            cm.validate_partition(&spec, &cm.get_partitions()).is_err(),
            "QoS not in allow_qos must fail validation"
        );

        // QoS in allow list → accepted.
        let mut spec = basic_spec("good-qos");
        spec.partition = Some("restricted".into());
        spec.qos = Some("premium".into());
        assert!(
            cm.validate_partition(&spec, &cm.get_partitions()).is_ok(),
            "QoS in allow_qos must pass validation"
        );

        // No QoS with allow_qos set → rejected (empty string not in list).
        let mut spec = basic_spec("no-qos");
        spec.partition = Some("restricted".into());
        spec.qos = None;
        assert!(
            cm.validate_partition(&spec, &cm.get_partitions()).is_err(),
            "absent QoS must fail when allow_qos is non-empty"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn validate_partition_rejects_qos_in_deny_qos() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let part = spur_core::partition::Partition {
            name: "nodebug".into(),
            nodes: "ALL".into(),
            deny_qos: vec!["debug".into()],
            ..Default::default()
        };
        cm.create_partition(part).unwrap();
        wait_for("nodebug created", || {
            cm.get_partitions().iter().any(|p| p.name == "nodebug")
        });

        // Denied QoS → rejected.
        let mut spec = basic_spec("denied-qos");
        spec.partition = Some("nodebug".into());
        spec.qos = Some("debug".into());
        assert!(
            cm.validate_partition(&spec, &cm.get_partitions()).is_err(),
            "denied QoS must fail validation"
        );

        // Non-denied QoS → allowed.
        let mut spec = basic_spec("other-qos");
        spec.partition = Some("nodebug".into());
        spec.qos = Some("premium".into());
        assert!(
            cm.validate_partition(&spec, &cm.get_partitions()).is_ok(),
            "non-denied QoS must pass validation"
        );

        // None QoS → allowed (deny_qos only blocks explicitly-named values).
        let mut spec = basic_spec("no-qos");
        spec.partition = Some("nodebug".into());
        spec.qos = None;
        assert!(
            cm.validate_partition(&spec, &cm.get_partitions()).is_ok(),
            "absent QoS must not be blocked by deny_qos"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn validate_partition_passes_any_qos_when_allow_qos_empty() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;

        let part = spur_core::partition::Partition {
            name: "open".into(),
            nodes: "ALL".into(),
            allow_qos: vec![],
            ..Default::default()
        };
        cm.create_partition(part).unwrap();
        wait_for("open created", || {
            cm.get_partitions().iter().any(|p| p.name == "open")
        });

        let mut spec = basic_spec("any-qos");
        spec.partition = Some("open".into());
        spec.qos = Some("whatever".into());
        assert!(
            cm.validate_partition(&spec, &cm.get_partitions()).is_ok(),
            "empty allow_qos must not restrict any QoS"
        );
    }

    // Full submit path: a user with no explicit `-q` whose association default
    // QoS is in the partition's allow_qos must be admitted. The tests above call
    // validate_partition with an explicit qos, so none covers this ordering.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn submit_admits_association_default_qos_on_allow_qos_partition() {
        let dir = TempDir::new().unwrap();
        let cm = test_cluster(&dir).await;
        cm.qos_cache().insert(Qos {
            name: "premium".into(),
            ..Default::default()
        });
        cm.association_cache()
            .insert_default_qos("testuser", "research", "premium");

        let part = spur_core::partition::Partition {
            name: "restricted".into(),
            nodes: "ALL".into(),
            allow_qos: vec!["premium".into()],
            ..Default::default()
        };
        cm.create_partition(part).unwrap();
        wait_for("restricted created", || {
            cm.get_partitions().iter().any(|p| p.name == "restricted")
        });

        let mut spec = basic_spec("inherits-default");
        spec.account = Some("research".into());
        spec.partition = Some("restricted".into());
        spec.qos = None;
        let id = cm
            .submit_job(spec)
            .expect("association default QoS must satisfy the partition allow_qos")
            .job_id;
        assert_eq!(
            cm.get_job(id).unwrap().spec.qos.as_deref(),
            Some("premium"),
            "the resolved default QoS must be recorded on the job"
        );
    }
}
