// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

use crate::partition::{Partition, PartitionState, PreemptMode, PreemptType};

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse TOML: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("missing required field: {0}")]
    MissingField(String),
    #[error("invalid value for {field}: {value}")]
    InvalidValue { field: String, value: String },
}

/// Top-level configuration (slurm.conf equivalent, in TOML).
///
/// We support reading both our native TOML format and the traditional
/// Slurm key=value format for migration purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlurmConfig {
    pub cluster_name: String,

    #[serde(default)]
    pub controller: ControllerConfig,

    #[serde(default)]
    pub accounting: AccountingConfig,

    #[serde(default)]
    pub scheduler: SchedulerConfig,

    #[serde(default)]
    pub auth: AuthConfig,

    #[serde(default)]
    pub partitions: Vec<PartitionConfig>,

    #[serde(default)]
    pub nodes: Vec<NodeConfig>,

    #[serde(default)]
    pub network: NetworkConfig,

    #[serde(default)]
    pub logging: LoggingConfig,

    #[serde(default)]
    pub kubernetes: KubernetesConfig,

    /// Native cluster (k0s) that SPUR provisions and owns. Inverse of `[kubernetes]`.
    #[serde(default)]
    pub cluster: ClusterConfig,

    #[serde(default)]
    pub notifications: NotificationConfig,

    #[serde(default)]
    pub power: PowerConfig,

    #[serde(default)]
    pub federation: FederationConfig,

    /// Topology configuration (switch hierarchy for locality-aware scheduling).
    #[serde(default)]
    pub topology: Option<crate::topology::TopologyConfig>,

    /// Job isolation configuration.
    #[serde(default)]
    pub isolation: IsolationConfig,

    /// Cluster-wide license pool, e.g., {"fluent": 20, "comsol": 5}.
    #[serde(default)]
    pub licenses: HashMap<String, u64>,

    /// Cluster-wide burst-buffer capacity pool.
    #[serde(default)]
    pub burst_buffer: BurstBufferConfig,

    /// Auto-update configuration.
    #[serde(default)]
    pub update: UpdateConfig,

    /// OpenMetrics HTTP export (spurctld, default port 6822).
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// REST API (Slurm-compatible HTTP, default port 6820).
    #[serde(default)]
    pub rest_api: RestApiConfig,

    /// Prolog/epilog hook scripts.
    #[serde(default)]
    pub hooks: HooksConfig,

    /// Device discovery, CDI, and GRES configuration.
    #[serde(default)]
    pub devices: DevicesConfig,

    /// Node admission control.
    #[serde(default)]
    pub admission: AdmissionConfig,

    /// POSIX per-process resource limits applied to job steps at launch.
    #[serde(default)]
    pub rlimits: RlimitsConfig,

    /// cgroup-v2 job resource enforcement (spurd, native-host jobs).
    #[serde(default)]
    pub cgroup: CgroupConfig,

    /// MPI plugin settings for `--mpi=pmix` job steps.
    #[serde(default)]
    pub mpi: MpiConfig,

    /// Node health-check program, run before a node re-enters the schedulable
    /// pool and on an interval; a failure drains the node (spurd).
    #[serde(default)]
    pub health: HealthConfig,
}

/// Configuration for auto-update checking and self-update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfig {
    /// Check for updates on daemon startup (default: true).
    #[serde(default = "default_true_fn")]
    pub check_on_startup: bool,

    /// Automatically download and install updates (default: false).
    /// Even when true, daemons will NOT auto-restart.
    #[serde(default)]
    pub auto_update: bool,

    /// Release channel: "stable" or "nightly" (default: "stable").
    #[serde(default = "default_stable")]
    pub channel: String,

    /// Directory for the update check cache file.
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
}

fn default_stable() -> String {
    "stable".into()
}
fn default_cache_dir() -> String {
    "/var/cache/spur".into()
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            check_on_startup: true,
            auto_update: false,
            channel: "stable".into(),
            cache_dir: "/var/cache/spur".into(),
        }
    }
}

/// OpenMetrics export settings for spurctld (separate listener from gRPC 6817).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MetricsConfig {
    /// When false, spurctld does not start the metrics HTTP server.
    #[serde(default = "default_true_fn")]
    pub enabled: bool,
    /// Metrics HTTP listen address (port used when `bind = "loopback"`).
    #[serde(default = "default_metrics_listen_addr")]
    pub listen_addr: String,
    /// `loopback` binds 127.0.0.1; `all` uses `listen_addr` as-is.
    #[serde(default)]
    pub bind: MetricsBind,
    /// Serves `/metrics/jobs-users-accts`, which 404s while false. Off by default
    /// because each job, user, and account becomes its own series.
    #[serde(default)]
    pub high_cardinality: bool,
}

fn default_metrics_listen_addr() -> String {
    "[::]:6822".into()
}

/// Metrics HTTP bind policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MetricsBind {
    #[default]
    Loopback,
    All,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            listen_addr: default_metrics_listen_addr(),
            bind: MetricsBind::Loopback,
            high_cardinality: false,
        }
    }
}

impl MetricsConfig {
    /// Listen socket after applying [`MetricsBind`].
    ///
    /// Returns an error if `listen_addr` is not a valid `SocketAddr`.
    pub fn effective_listen_addr(&self) -> Result<std::net::SocketAddr, ConfigError> {
        let addr = self
            .listen_addr
            .parse::<std::net::SocketAddr>()
            .map_err(|e| ConfigError::InvalidValue {
                field: "metrics.listen_addr".into(),
                value: e.to_string(),
            })?;
        Ok(match self.bind {
            MetricsBind::All => addr,
            MetricsBind::Loopback => std::net::SocketAddr::from(([127, 0, 0, 1], addr.port())),
        })
    }
}

/// REST API settings for spurctld (Slurm-compatible HTTP on a separate port).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RestApiConfig {
    /// When false, spurctld does not start the REST server.
    ///
    /// Defaults to FALSE: the REST surface performs no authentication, and its submit handler builds
    /// a job spec directly from the request body, so enabling it on a reachable address exposes
    /// unauthenticated job submission. Enable it only behind an authenticating proxy or on a
    /// loopback/administrative interface.
    #[serde(default)]
    pub enabled: bool,
}

/// Prolog and epilog hook script configuration.
///
/// All fields are optional — `None` means no hook is configured for that point.
/// Paths must be fully qualified; no search path is set for security reasons.
/// Hook lifecycle and failure semantics match Slurm.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HooksConfig {
    /// Script run on compute nodes before job launch (Slurm `Prolog`).
    pub prolog: Option<String>,
    /// Script run on compute nodes at job termination (Slurm `Epilog`).
    pub epilog: Option<String>,
    /// Script run on the controller at job allocation (Slurm `PrologSlurmctld`).
    pub prolog_slurmctld: Option<String>,
    /// Script run on the controller at job termination (Slurm `EpilogSlurmctld`).
    pub epilog_slurmctld: Option<String>,
    /// Script run on compute nodes before each job step (Slurm `TaskProlog`).
    pub task_prolog: Option<String>,
    /// Script run on compute nodes after each job step (Slurm `TaskEpilog`).
    pub task_epilog: Option<String>,
    /// Script run on the srun invocation node before step dispatch (Slurm `SrunProlog`).
    pub srun_prolog: Option<String>,
    /// Script run on the srun invocation node after step completion (Slurm `SrunEpilog`).
    pub srun_epilog: Option<String>,
    /// Script run on the controller at submission to accept/reject/modify a job.
    /// Receives the resolved spec as JSON on stdin (Slurm `job_submit.lua` analog).
    pub job_submit: Option<String>,
    /// Lua script defining `slurm_job_submit(job_desc, submit_uid)`, run in a
    /// sandbox at submission. Slurm `job_submit/lua` parity. Runs after `job_submit`.
    pub job_submit_lua: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerConfig {
    /// gRPC listen address for the controller.
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,
    /// REST API listen address.
    #[serde(default = "default_rest_addr")]
    pub rest_addr: String,
    /// Hostname(s) for HA. First is primary.
    #[serde(default = "default_hosts")]
    pub hosts: Vec<String>,
    /// State save location.
    #[serde(default = "default_state_dir")]
    pub state_dir: String,
    /// Max job ID before wrapping.
    #[serde(default = "default_max_job_id")]
    pub max_job_id: u32,
    /// First job ID.
    #[serde(default = "default_one")]
    pub first_job_id: u32,

    /// Raft peers for HA consensus, "host:port" each. Empty = single-node mode.
    /// Must be identically ordered on every controller: node ids derive from
    /// list position, so a reordered list can form an inconsistent voter set.
    /// Example: ["node1:6821", "node2:6821", "node3:6821"]
    #[serde(default)]
    pub peers: Vec<String>,

    /// This node's Raft ID. Normally unset; single-node mode always uses 1.
    /// Otherwise resolved as: explicit value, else position in `peers` (by
    /// hostname), else hostname ordinal (IP-only peers). Must be in
    /// `1..=peers.len()` and, if set, equal its position (index + 1).
    pub node_id: Option<u64>,

    /// Listen address for Raft internal gRPC traffic (separate from client API).
    #[serde(default = "default_raft_listen_addr")]
    pub raft_listen_addr: String,

    /// Seconds without a heartbeat before a node is marked Down.
    /// Defaults to 90 when absent.
    #[serde(default)]
    pub heartbeat_timeout_secs: Option<u64>,

    /// Maximum automatic requeues (excluding preemption) before a job is held
    /// with `JobHoldMaxRequeue`. Configured in TOML as `[controller] max_batch_requeue` (default: 5).
    #[serde(default = "default_max_batch_requeue")]
    pub max_batch_requeue: u32,

    /// Upper bound on the hold placed between automatic requeues after a launch
    /// failure. The hold doubles per attempt, so this caps how long a job with a
    /// large `max_batch_requeue` can wait between retries. Configured in TOML as
    /// `[controller] max_launch_backoff_secs` (default: 300).
    #[serde(default = "default_max_launch_backoff_secs")]
    pub max_launch_backoff_secs: u64,

    /// Hold a job at priority 0 when its prolog fails, instead of requeueing it
    /// for another attempt. On by default: a prolog receives the job's own
    /// context, so the same failure recurs on every node, and retrying would
    /// drain one node per attempt. Turning it off restores the retry, bounded by
    /// `max_batch_requeue`. Configured in TOML as
    /// `[controller] hold_on_prolog_fail` (default: true). Slurm's equivalent is
    /// the inverse `SchedulerParameters=nohold_on_prolog_fail`.
    #[serde(default = "default_hold_on_prolog_fail")]
    pub hold_on_prolog_fail: bool,

    /// Seconds a terminal job stays in controller memory before eviction (default
    /// 3600). `sacct` history is unaffected but `scontrol show job` stops finding
    /// it after the window; runs once per `scheduler.interval_secs` and is floored
    /// to the accounting reconcile interval so a job's DB row can be repaired first.
    #[serde(default = "default_terminal_job_retention_secs")]
    pub terminal_job_retention_secs: u64,

    /// Seconds a node is skipped for new dispatch after rejecting a launch or failing to
    /// be reached, so it isn't re-picked every tick (default 30, 0 disables).
    #[serde(default = "default_dispatch_reject_cooldown_secs")]
    pub dispatch_reject_cooldown_secs: u64,

    /// Ceiling on a single launch or allocation-register RPC to an agent (default 300, 0 disables).
    /// Must exceed the slowest node prolog, which runs inside the launch call.
    #[serde(default = "default_dispatch_timeout_secs")]
    pub dispatch_timeout_secs: u64,

    /// Budget for establishing a controller-to-agent connection (default 5, 0 falls back to the
    /// OS TCP timeout).
    #[serde(default = "default_agent_connect_timeout_secs")]
    pub agent_connect_timeout_secs: u64,

    /// HTTP/2 ping interval on an open agent connection (default 10, 0 disables keepalive).
    /// Detection of a silent peer takes roughly this plus `agent_keepalive_timeout_secs`.
    #[serde(default = "default_agent_keepalive_interval_secs")]
    pub agent_keepalive_interval_secs: u64,

    /// How long to wait for a ping response before dropping the connection (default 10).
    #[serde(default = "default_agent_keepalive_timeout_secs")]
    pub agent_keepalive_timeout_secs: u64,

    /// How much of another user's job a non-owner may see via `get_job` /
    /// `get_job_steps`. See [`JobInfoVisibility`]. Owners and admins always see
    /// the full record; this governs everyone else. Default: `redacted`.
    #[serde(default)]
    pub job_info_visibility: JobInfoVisibility,
}

/// Controls how much of another user's job a non-owner (non-admin) caller can
/// read back from `get_job` / `get_job_steps`.
///
/// The list RPC `get_jobs` already scopes to the caller; the single-fetch paths
/// historically did not, exposing every job's work_dir, command line, stdio
/// paths, and — most usefully to an attacker — its allocated nodelist. This
/// setting closes that leak while leaving the Slurm-standard cluster-visible
/// queue intact for the fields that are not targeting-sensitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum JobInfoVisibility {
    /// Non-owners see identity/state/timing/account/priority, but work_dir,
    /// command, stdio paths, allocated nodelist, comment, and resource detail are
    /// blanked. The default: preserves visibility, removes the targeting oracle.
    #[default]
    Redacted,
    /// Non-owners get `NOT_FOUND` — the job is invisible unless you own it (or
    /// are an admin). Strictest; matches `get_jobs`' owner-scoped behaviour.
    OwnerOnly,
    /// Legacy: every field is visible to any caller. Opt-in for deployments that
    /// relied on the previous unscoped behaviour.
    Full,
}

fn default_max_batch_requeue() -> u32 {
    5
}

fn default_terminal_job_retention_secs() -> u64 {
    3600
}

fn default_dispatch_reject_cooldown_secs() -> u64 {
    30
}

fn default_dispatch_timeout_secs() -> u64 {
    300
}

/// Shared with `spurctld`'s agent channel builder, which needs these before any config is loaded.
/// A dial budget or ping interval beyond this is indistinguishable from disabling it.
pub const MAX_AGENT_CHANNEL_TIMEOUT_SECS: u64 = 600;

pub const DEFAULT_AGENT_CONNECT_TIMEOUT_SECS: u64 = 5;
pub const DEFAULT_AGENT_KEEPALIVE_INTERVAL_SECS: u64 = 10;
pub const DEFAULT_AGENT_KEEPALIVE_TIMEOUT_SECS: u64 = 10;

fn default_agent_connect_timeout_secs() -> u64 {
    DEFAULT_AGENT_CONNECT_TIMEOUT_SECS
}

fn default_agent_keepalive_interval_secs() -> u64 {
    DEFAULT_AGENT_KEEPALIVE_INTERVAL_SECS
}

fn default_agent_keepalive_timeout_secs() -> u64 {
    DEFAULT_AGENT_KEEPALIVE_TIMEOUT_SECS
}

fn default_hold_on_prolog_fail() -> bool {
    true
}

fn default_max_launch_backoff_secs() -> u64 {
    300
}

/// Upper bound for `max_launch_backoff_secs`. A per-attempt retry hold beyond a
/// day is not a retry policy, and larger values push the computed hold instant
/// out of chrono's representable range.
pub const MAX_LAUNCH_BACKOFF_SECS: u64 = 86_400;

/// Upper bound for `terminal_job_retention_secs` (one year). Operational
/// guardrail: retaining terminal jobs longer defeats the memory bound.
pub const MAX_TERMINAL_JOB_RETENTION_SECS: u64 = 366 * 24 * 60 * 60;

/// Upper bound for `grp_wall_window_days` (ten years). The window is bound into
/// the consumption query as an `i32`, so anything that could wrap there must be
/// rejected first: a negative interval moves the cutoff into the future and the
/// query matches nothing, leaving every `GrpWall` budget unenforced.
pub const MAX_GRP_WALL_WINDOW_DAYS: u32 = 3650;

fn default_listen_addr() -> String {
    "[::]:6817".into()
}
fn default_raft_listen_addr() -> String {
    "[::]:6821".into()
}
fn default_rest_addr() -> String {
    "[::]:6820".into()
}
fn default_hosts() -> Vec<String> {
    vec!["localhost".into()]
}
fn default_state_dir() -> String {
    "/var/spool/spur".into()
}
fn default_max_job_id() -> u32 {
    999_999_999
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            listen_addr: "[::]:6817".into(),
            rest_addr: "[::]:6820".into(),
            hosts: vec!["localhost".into()],
            state_dir: "/var/spool/spur".into(),
            max_job_id: 999_999_999,
            first_job_id: 1,
            peers: Vec::new(),
            node_id: None,
            raft_listen_addr: "[::]:6821".into(),
            heartbeat_timeout_secs: None,
            max_batch_requeue: default_max_batch_requeue(),
            max_launch_backoff_secs: default_max_launch_backoff_secs(),
            hold_on_prolog_fail: default_hold_on_prolog_fail(),
            terminal_job_retention_secs: default_terminal_job_retention_secs(),
            dispatch_reject_cooldown_secs: default_dispatch_reject_cooldown_secs(),
            dispatch_timeout_secs: default_dispatch_timeout_secs(),
            agent_connect_timeout_secs: default_agent_connect_timeout_secs(),
            agent_keepalive_interval_secs: default_agent_keepalive_interval_secs(),
            agent_keepalive_timeout_secs: default_agent_keepalive_timeout_secs(),
            job_info_visibility: JobInfoVisibility::default(),
        }
    }
}

impl ControllerConfig {
    /// Client-facing controller endpoints, one per `hosts` entry, each combined
    /// with the port from `listen_addr`. Used for client-side failover across an
    /// HA quorum.
    pub fn endpoints(&self) -> Vec<String> {
        let port = self.listen_addr.rsplit(':').next().unwrap_or("6817");
        self.hosts
            .iter()
            .map(|host| {
                if host.contains(':') && !host.starts_with('[') {
                    format!("http://[{host}]:{port}")
                } else {
                    format!("http://{host}:{port}")
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountingConfig {
    /// PostgreSQL connection string for accounting. When non-empty, spurctld
    /// connects and serves the SlurmAccounting gRPC service on port 6817
    /// alongside the controller API. Empty (default) disables accounting.
    #[serde(default)]
    pub database_url: String,
    /// How often to refresh fairshare/QoS caches from the accounting database.
    #[serde(default = "default_fairshare_refresh_secs")]
    pub fairshare_refresh_secs: u32,
    /// Trailing window over which a QOS's wall-clock consumption is measured for
    /// `GrpWall`. Independent of `scheduler.fairshare_halflife_days`: that fades
    /// old usage for priority scoring, while this is a hard cutoff on a budget, so
    /// the two are tuned for different reasons and are expected to diverge.
    #[serde(default = "default_grp_wall_window_days")]
    pub grp_wall_window_days: u32,
    /// Cluster-wide fallback QOS, applied at submit when a job resolves to no
    /// QOS. The last link in the resolution chain (Slurm's stock `normal`
    /// analogue). Empty (default) = no fallback.
    #[serde(default)]
    pub default_qos: String,
    /// Reject at submit any job that still has no QOS after the resolution
    /// chain. Mirrors Slurm's `AccountingStorageEnforce=qos`. Default false.
    #[serde(default)]
    pub require_qos: bool,
    /// Reject at submit any job whose user resolves to no account at all.
    /// Mirrors Slurm's `AccountingStorageEnforce=associations`. Default false.
    #[serde(default)]
    pub require_association: bool,
    /// Delete `txn` audit-log rows older than this many days. `None` (default)
    /// or `0` disables purging (rows kept forever, matching Slurm's default-off
    /// behavior); a positive value enables the periodic purge.
    #[serde(default)]
    pub txn_retention_days: Option<u32>,
}

impl AccountingConfig {
    /// True when accounting is configured (a database URL is set). When false there is no
    /// association database to consult, so account/association checks have nothing to enforce.
    pub fn enabled(&self) -> bool {
        !self.database_url.is_empty()
    }
}

fn default_fairshare_refresh_secs() -> u32 {
    300
}

fn default_grp_wall_window_days() -> u32 {
    14
}

impl Default for AccountingConfig {
    fn default() -> Self {
        Self {
            database_url: String::new(),
            fairshare_refresh_secs: 300,
            grp_wall_window_days: default_grp_wall_window_days(),
            default_qos: String::new(),
            require_qos: false,
            require_association: false,
            txn_retention_days: None,
        }
    }
}

/// Submit-time enforcement of partition wall-time (Slurm `EnforcePartLimits`):
/// `No` admits over-limit jobs to pend, `All` needs all fit, `Any` needs one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum EnforcePartLimits {
    #[default]
    No,
    All,
    Any,
}

// Case-insensitive, and accepts Slurm's `YES` alias for `ALL`, so a value like
// `enforce_part_limits = "all"` in spur.conf does not fail controller startup.
impl<'de> Deserialize<'de> for EnforcePartLimits {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.trim().to_ascii_uppercase().as_str() {
            "NO" => Ok(Self::No),
            "ALL" | "YES" => Ok(Self::All),
            "ANY" => Ok(Self::Any),
            other => Err(serde::de::Error::custom(format!(
                "invalid enforce_part_limits '{other}' (expected NO, ALL, or ANY)"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler plugin name.
    #[serde(default = "default_scheduler_plugin")]
    pub plugin: String,
    /// How often to run the scheduler (seconds).
    #[serde(default = "default_one")]
    pub interval_secs: u32,
    /// Max jobs to evaluate per cycle.
    #[serde(default = "default_max_jobs")]
    pub max_jobs_per_cycle: u32,
    /// Fair-share decay half-life (days).
    #[serde(default = "default_halflife")]
    pub fairshare_halflife_days: u32,
    /// Cluster-wide fallback wall-time (minutes) for a `-t`-less job on a partition
    /// with no `DefaultTime`. `0` (default) leaves such jobs unbounded; > 0 bounds them.
    #[serde(default = "default_time_limit")]
    pub default_time_limit_minutes: u32,
    /// Whether to reject over-limit jobs at submit (Slurm `EnforcePartLimits`).
    #[serde(default)]
    pub enforce_part_limits: EnforcePartLimits,
    /// Max seconds to wait in COMPLETING before force-finishing the job.
    #[serde(default = "default_complete_wait")]
    pub complete_wait_secs: u32,
    /// Grace minutes after a reservation ends before cancelling its running jobs.
    #[serde(default)]
    pub resv_overrun_minutes: u32,
    /// Reap an interactive allocation (salloc/srun) whose client has sent no
    /// keepalive for this many seconds. `0` (the default) disables reaping, so
    /// abandoned allocations behave as before. Mirrors Slurm's `InactiveLimit`.
    #[serde(default)]
    pub inactive_limit_secs: u32,
    /// Highest base priority a non-admin caller may request at submit (or via
    /// `scontrol update`). Requests above this are clamped, not rejected; at
    /// submit the clamp is reported to the caller as a warning, on the update
    /// path (which has no response field) it is logged. Admins are unaffected.
    /// Defaults to [`crate::job::DEFAULT_PRIORITY`], so a non-admin can lower but
    /// not raise priority — Slurm's `nice`-only model, where boosting priority is
    /// operator-only. Raise it to grant users a band above the baseline.
    #[serde(default = "default_max_user_priority")]
    pub max_user_priority: u32,
    /// Controls which jobs are eligible to preempt which. `None` (default)
    /// enforces no cross-QOS restrictions — any job with sufficient priority gap
    /// may preempt any other. `QosPriority` requires the pending job's QOS to
    /// list the running job's QOS in its `preempt` allow-list. Mirrors Slurm's
    /// `PreemptType`.
    #[serde(default)]
    pub preempt_type: PreemptType,
    /// Cluster-wide minimum number of seconds a job must have been running before
    /// it becomes eligible for preemption. Can be overridden per-partition and
    /// per-QOS. `0` (default) means immediately eligible. Mirrors Slurm's
    /// `PreemptExemptTime`.
    #[serde(default)]
    pub preempt_exempt_time: u32,
}

/// How often an interactive client (`salloc`/`srun`) pings the controller to
/// keep its allocation attended. Shared with the CLI so `inactive_limit_secs`
/// can be validated against it.
pub const KEEPALIVE_INTERVAL_SECS: u64 = 30;

fn default_scheduler_plugin() -> String {
    "backfill".into()
}
fn default_max_jobs() -> u32 {
    10000
}
fn default_halflife() -> u32 {
    14
}
fn default_time_limit() -> u32 {
    0
}
fn default_complete_wait() -> u32 {
    300
}
fn default_max_user_priority() -> u32 {
    crate::job::DEFAULT_PRIORITY
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            plugin: "backfill".into(),
            interval_secs: 1,
            max_jobs_per_cycle: 10000,
            fairshare_halflife_days: 14,
            default_time_limit_minutes: 0,
            enforce_part_limits: EnforcePartLimits::No,
            complete_wait_secs: 300,
            resv_overrun_minutes: 0,
            inactive_limit_secs: 0,
            max_user_priority: default_max_user_priority(),
            preempt_type: PreemptType::None,
            preempt_exempt_time: 0,
        }
    }
}

/// How strictly the control plane authenticates its callers.
///
/// Exists so a running cluster can adopt authentication without an outage: bring the control plane
/// up in `permissive`, roll credentials out to clients (the logs name every caller still
/// unauthenticated), then flip to `required`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AuthMode {
    /// Do not authenticate at all; trust the identity asserted by the client.
    Disabled,
    /// Verify a credential when one is presented, and reject it if invalid; fall back to the
    /// client-asserted identity when none is presented, logging the caller. The default, because
    /// flipping an existing cluster straight to `required` locks out every client at once.
    #[default]
    Permissive,
    /// Require a valid credential on every request.
    Required,
}

impl AuthMode {
    pub fn as_str(self) -> &'static str {
        match self {
            AuthMode::Disabled => "disabled",
            AuthMode::Permissive => "permissive",
            AuthMode::Required => "required",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Auth plugin: "jwt" (implemented) or "none". "munge" is recognised but not implemented and is
    /// rejected at startup rather than silently ignored.
    pub plugin: String,
    /// How strictly callers are authenticated. See [`AuthMode`].
    #[serde(default)]
    pub mode: AuthMode,
    /// HMAC signing secret (inline), signing and verifying both user credentials
    /// (`spur token user`) and node admission tokens; used literally, and the
    /// resolved key is captured at startup, not updated by `reconfigure`.
    pub jwt_key: Option<String>,
    /// Path to a regular file containing the JWT secret key.
    pub jwt_key_file: Option<String>,
    /// Allow jobs to execute as uid 0 (root).
    ///
    /// Default false, and deliberately so: `uid` arrives on the wire as part of the job spec, and a
    /// job requesting uid 0 previously *skipped* the privilege drop rather than failing it, so any
    /// peer that could reach the controller could obtain root on a compute node. Enable only on a
    /// cluster where every submitter is already trusted with root.
    #[serde(default)]
    pub allow_root_jobs: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            plugin: "jwt".into(),
            mode: AuthMode::default(),
            jwt_key: None,
            jwt_key_file: None,
            allow_root_jobs: false,
        }
    }
}

impl AuthConfig {
    /// Resolve the signing key once at process startup.
    ///
    /// `jwt_key` is always treated as a literal secret. `jwt_key_file` is the explicit
    /// file-backed form. A trailing line ending in the file is ignored so a file created with
    /// `echo` has the same secret as its inline form.
    pub fn resolved_jwt_key(&self) -> Result<Option<String>, ConfigError> {
        if self.jwt_key.is_some() && self.jwt_key_file.is_some() {
            return Err(ConfigError::InvalidValue {
                field: "auth".into(),
                value: "jwt_key and jwt_key_file cannot both be set".into(),
            });
        }

        if let Some(value) = &self.jwt_key {
            if value.trim().is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "auth.jwt_key".into(),
                    value: "must not be blank".into(),
                });
            }
            return Ok(Some(value.clone()));
        }

        let Some(path) = &self.jwt_key_file else {
            return Ok(None);
        };

        let path = Path::new(path);
        let metadata = std::fs::metadata(path)?;
        if !metadata.is_file() {
            return Err(ConfigError::InvalidValue {
                field: "auth.jwt_key_file".into(),
                value: format!("{} is not a regular file", path.display()),
            });
        }

        let key = std::fs::read_to_string(path)?;
        let key = key.trim_end_matches(['\r', '\n']).to_owned();
        if key.trim().is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "auth.jwt_key_file".into(),
                value: "contains a blank signing key".into(),
            });
        }
        Ok(Some(key))
    }

    pub fn has_jwt_key(&self) -> bool {
        match self.resolved_jwt_key() {
            Ok(key) => key.is_some(),
            Err(error) => {
                tracing::warn!(%error, "failed to resolve jwt key; treating as not configured");
                false
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    pub name: String,
    #[serde(default)]
    pub default: bool,
    #[serde(default = "default_partition_state")]
    pub state: String,
    #[serde(default)]
    pub nodes: String,
    #[serde(default)]
    pub selector: HashMap<String, String>,
    pub max_time: Option<String>,
    pub default_time: Option<String>,
    pub max_nodes: Option<u32>,
    #[serde(default = "default_one")]
    pub min_nodes: u32,
    #[serde(default)]
    pub allow_accounts: Vec<String>,
    #[serde(default)]
    pub allow_groups: Vec<String>,
    #[serde(default)]
    pub deny_accounts: Vec<String>,
    #[serde(default)]
    pub deny_qos: Vec<String>,
    #[serde(default)]
    pub allow_qos: Vec<String>,
    #[serde(default)]
    pub priority_tier: u32,
    #[serde(default)]
    pub preempt_mode: String,
    /// Minimum seconds a job must have been running before it is eligible for
    /// preemption from this partition. Overrides the cluster-wide default.
    /// `None` defers to `scheduler.preempt_exempt_time`.
    #[serde(default)]
    pub preempt_exempt_time: Option<u32>,
}

fn default_partition_state() -> String {
    "UP".into()
}
fn default_one() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Hostlist pattern for this node definition. Optional when selector is used.
    #[serde(default)]
    pub names: String,
    /// Label selector: apply this config to nodes matching ALL key-value pairs.
    #[serde(default)]
    pub selector: HashMap<String, String>,
    #[serde(default)]
    pub cpus: u32,
    #[serde(default)]
    pub memory_mb: u64,
    #[serde(default)]
    pub gres: Vec<String>,
    #[serde(default)]
    pub features: Vec<String>,
    /// Default comm address when the agent has not registered one yet. Does not
    /// override an agent-registered address.
    pub address: Option<String>,
    /// Scheduling weight. Higher weight = preferred for scheduling.
    #[serde(default = "default_one")]
    pub weight: u32,
}

/// Network / WireGuard mesh configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Enable WireGuard mesh networking between nodes.
    #[serde(default)]
    pub wg_enabled: bool,
    /// CIDR block for WireGuard address allocation (default: 10.44.0.0/16).
    #[serde(default = "default_wg_cidr")]
    pub wg_cidr: String,
    /// WireGuard interface name (default: spur0).
    #[serde(default = "default_wg_interface")]
    pub wg_interface: String,
    /// WireGuard listen port (default: 51820).
    #[serde(default = "default_wg_port")]
    pub wg_port: u16,
    /// Agent gRPC listen port (default: 6818).
    #[serde(default = "default_agent_port")]
    pub agent_port: u16,
    /// Reject agent registrations whose comm address is not routable (loopback,
    /// unspecified, or link-local).
    #[serde(default)]
    pub reject_loopback_comm_addr: bool,
}

fn default_wg_cidr() -> String {
    "10.44.0.0/16".into()
}
fn default_wg_interface() -> String {
    "spur0".into()
}
fn default_wg_port() -> u16 {
    51820
}
fn default_agent_port() -> u16 {
    6818
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            wg_enabled: false,
            wg_cidr: "10.44.0.0/16".into(),
            wg_interface: "spur0".into(),
            wg_port: 51820,
            agent_port: 6818,
            reject_loopback_comm_addr: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub file: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "text".into(),
            file: None,
        }
    }
}

/// Kubernetes integration configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesConfig {
    /// Enable K8s integration.
    #[serde(default)]
    pub enabled: bool,
    /// Path to kubeconfig file. If empty, uses in-cluster config.
    #[serde(default)]
    pub kubeconfig: Option<String>,
    /// K8s namespace for SpurJob CRDs and Pods.
    #[serde(default = "default_k8s_namespace")]
    pub namespace: String,
    /// Label selector for K8s nodes to include in the Spur pool.
    #[serde(default = "default_k8s_node_selector")]
    pub node_label_selector: String,
}

fn default_k8s_namespace() -> String {
    "spur".into()
}

fn default_k8s_node_selector() -> String {
    "spur.amd.com/managed=true".into()
}

impl Default for KubernetesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            kubeconfig: None,
            namespace: "spur".into(),
            node_label_selector: "spur.amd.com/managed=true".into(),
        }
    }
}

/// Native cluster (k0s) integration — SPUR OWNS the Kubernetes cluster lifecycle
/// (`spur k8s up/down`, spurd-owned k0s systemd units, GPU CDI on join). This is the
/// inverse of `[kubernetes]` above (which lets SPUR run *inside* an existing k8s and accept
/// SpurJob CRDs); the two are intentionally distinct sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Enable SPUR-managed k0s. When false, spurd never touches systemd/k0s.
    #[serde(default)]
    pub enabled: bool,
    /// Kubernetes distribution SPUR manages. Only "k0s" is supported today.
    #[serde(default = "default_cluster_distro")]
    pub distro: String,
    /// Pod network CIDR. Calico bird native routing over the mesh carves per-node /24s from this.
    #[serde(default = "default_pod_cidr")]
    pub pod_cidr: String,
    /// Service network CIDR.
    #[serde(default = "default_service_cidr")]
    pub service_cidr: String,
    /// CNI MTU (Calico only). Defaults to 1450 to leave headroom for WireGuard's ~50-byte overhead
    /// over the mesh; set explicitly if your underlay differs. Emitted into the generated k0s config.
    #[serde(default = "default_cni_mtu")]
    pub cni_mtu: u16,
    /// Hostname of the node that runs the k0s control plane. Empty = pick from inventory.
    #[serde(default)]
    pub control_plane_node: Option<String>,
    /// HA control-plane count (1, 3, or 5). Overridden per-invocation by `spur k8s up --replicas`.
    #[serde(default = "default_control_plane_replicas")]
    pub control_plane_replicas: u32,
    /// k0s release to install/run (e.g. "v1.36.2+k0s.0", or "latest"). Pinned to a known-good
    /// version by default; bumped per spur release. spurd installs this if the binary is missing.
    #[serde(default = "default_k0s_version")]
    pub k0s_version: String,
    /// Filesystem path to the k0s binary (install target + what the systemd unit runs).
    #[serde(default = "default_k0s_binary")]
    pub k0s_binary: String,
    /// CNI / network mode. "kuberouter" (k0s default — no custom config) or "calico" (Calico in
    /// bird native-routing mode with the API advertised on the mesh IP, so pods route over the
    /// WireGuard mesh). Selecting "calico" makes `spur k8s up` generate the k0s config + set each
    /// worker's kubelet `--node-ip` to its mesh IP.
    #[serde(default = "default_cni")]
    pub cni: String,
    /// Storage provisioner SPUR ships so PVC workloads work out of the box (k0s bundles none).
    /// "local-path" (default — RWO node-local, set as the default StorageClass) or "none" (bring your
    /// own). Applied via k0s's manifest deployer on the control-plane node.
    #[serde(default = "default_storage_provisioner")]
    pub storage_provisioner: String,
    /// On-node directory the local-path provisioner stores PersistentVolumes in. Point this at a
    /// large scratch disk if PVCs will hold much data — the default lives under `/var/lib` (root fs).
    #[serde(default = "default_local_path_dir")]
    pub local_path_dir: String,
    /// Seconds a k8s (k0s) node may stay non-`active` during provisioning before the reconcile
    /// loop marks the cluster `degraded` (surfacing which nodes failed). Covers the ~262 MB k0s
    /// download plus multi-node join.
    #[serde(default = "default_k8s_provisioning_timeout_secs")]
    pub k8s_provisioning_timeout_secs: u64,
    /// Allow `spur k8s kubeconfig --admin` to serve the k0s CLUSTER-ADMIN kubeconfig over RPC.
    ///
    /// Default false. The admin check it sits behind compares a client-supplied caller string, so
    /// while user authentication is unenforced this RPC would hand a cluster-admin credential to any
    /// peer that can reach the controller. With this off, obtain the admin kubeconfig on the
    /// control-plane node itself (`k0s kubeconfig admin`); per-user scoped kubeconfigs are
    /// unaffected.
    #[serde(default)]
    pub allow_admin_kubeconfig: bool,
}

fn default_cluster_distro() -> String {
    "k0s".into()
}
fn default_k0s_version() -> String {
    crate::k0s::K0S_PINNED_VERSION.into()
}
fn default_control_plane_replicas() -> u32 {
    1
}
fn default_k0s_binary() -> String {
    crate::k0s::K0S_DEFAULT_BINARY.into()
}
fn default_cni() -> String {
    "kuberouter".into()
}
fn default_pod_cidr() -> String {
    "10.42.0.0/16".into()
}
fn default_service_cidr() -> String {
    "10.43.0.0/16".into()
}
fn default_cni_mtu() -> u16 {
    1450
}
fn default_storage_provisioner() -> String {
    "local-path".into()
}
fn default_local_path_dir() -> String {
    crate::k0s::DEFAULT_LOCAL_PATH_DIR.into()
}
fn default_k8s_provisioning_timeout_secs() -> u64 {
    600
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            distro: default_cluster_distro(),
            pod_cidr: default_pod_cidr(),
            service_cidr: default_service_cidr(),
            cni_mtu: default_cni_mtu(),
            control_plane_node: None,
            control_plane_replicas: default_control_plane_replicas(),
            k0s_version: default_k0s_version(),
            k0s_binary: default_k0s_binary(),
            cni: default_cni(),
            storage_provisioner: default_storage_provisioner(),
            local_path_dir: default_local_path_dir(),
            k8s_provisioning_timeout_secs: default_k8s_provisioning_timeout_secs(),
            allow_admin_kubeconfig: false,
        }
    }
}

/// Basic dependency-free CIDR sanity check (`<ip>/<prefix>`), used by config validation.
fn is_valid_cidr(s: &str) -> bool {
    // IPv4 only: the native k0s provisioning paths (cluster_k8s IPAM, pod-CIDR carving, AddressPool)
    // are all `Ipv4Addr`, so an IPv6 CIDR would validate here and then fail later at runtime.
    match s.split_once('/') {
        Some((ip, prefix)) => {
            ip.parse::<std::net::Ipv4Addr>().is_ok()
                && prefix.parse::<u8>().map(|p| p <= 32).unwrap_or(false)
        }
        None => false,
    }
}

/// Prefix length of a CIDR string. Returns 255 (an impossible prefix, so any `<=` bound rejects it)
/// when unparseable — callers gate with `is_valid_cidr` first, so this only extracts the number.
fn cidr_prefix(s: &str) -> u8 {
    s.split_once('/')
        .and_then(|(_, p)| p.parse().ok())
        .unwrap_or(255)
}

/// True if two IPv4 CIDRs overlap (share any address). Non-IPv4 / malformed inputs return false
/// (they are rejected separately by `is_valid_cidr`).
fn cidrs_overlap(a: &str, b: &str) -> bool {
    fn parse_v4(s: &str) -> Option<(u32, u8)> {
        let (ip, prefix) = s.split_once('/')?;
        let ip: std::net::Ipv4Addr = ip.parse().ok()?;
        let prefix: u8 = prefix.parse().ok()?;
        if prefix > 32 {
            return None;
        }
        Some((u32::from(ip), prefix))
    }
    let (Some((a_ip, a_pfx)), Some((b_ip, b_pfx))) = (parse_v4(a), parse_v4(b)) else {
        return false;
    };
    let shorter = a_pfx.min(b_pfx);
    let mask = if shorter == 0 {
        0
    } else {
        u32::MAX << (32 - shorter)
    };
    (a_ip & mask) == (b_ip & mask)
}

/// Power management configuration for suspending/resuming idle nodes.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerConfig {
    /// Seconds a node must be idle before it is suspended.
    pub suspend_timeout_secs: Option<u64>,
    /// Command to suspend a node (e.g., "systemctl suspend"). {node} is replaced with the node name.
    pub suspend_command: Option<String>,
    /// Command to resume a node (e.g., "ipmitool chassis power on"). {node} is replaced with the node name.
    pub resume_command: Option<String>,
}

/// Notification configuration for job event webhooks and email.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NotificationConfig {
    /// Webhook URL to POST job event notifications to.
    pub webhook_url: Option<String>,
    /// SMTP command for sending mail, e.g., "/usr/sbin/sendmail -t".
    pub smtp_command: Option<String>,
    /// From address for notification emails, e.g., "spur@cluster.local".
    pub from_address: Option<String>,
}

/// Cluster-wide burst-buffer capacity, modeled like a single consumable pool
/// (analogous to a license total). Jobs reserve GB via `--bb capacity=NNN`.
///
/// ```toml
/// [burst_buffer]
/// total_gb = 1024
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct BurstBufferConfig {
    /// Total burst-buffer capacity in gibibytes. 0 (default) disables BB: any
    /// job requesting capacity stays pending with `BurstBufferResources`.
    #[serde(default)]
    pub total_gb: u64,
}

/// Job isolation configuration for native-host and container jobs.
///
/// Each layer operates independently and degrades gracefully when the
/// kernel doesn't support it or spurd isn't running as root.
///
/// Example:
/// ```toml
/// [isolation]
/// setuid = true       # Run jobs as submitting user (requires root)
/// namespaces = true   # PID + mount namespace isolation
/// seccomp = true      # syscall whitelist (blocks ptrace, mount, bpf)
/// landlock = true     # filesystem access control (kernel 5.13+)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationConfig {
    /// Run jobs as the submitting user's UID/GID (requires root spurd).
    #[serde(default = "default_true_fn")]
    pub setuid: bool,
    /// PID + mount namespace isolation (requires root).
    #[serde(default = "default_true_fn")]
    pub namespaces: bool,
    /// seccomp-BPF syscall filter (kernel 3.5+).
    #[serde(default = "default_true_fn")]
    pub seccomp: bool,
    /// Landlock filesystem access control (kernel 5.13+, native-host only).
    #[serde(default = "default_true_fn")]
    pub landlock: bool,
}

fn default_true_fn() -> bool {
    true
}

impl Default for IsolationConfig {
    fn default() -> Self {
        Self {
            setuid: true,
            namespaces: true,
            seccomp: true,
            landlock: true,
        }
    }
}

/// Federation configuration for multi-cluster job routing.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FederationConfig {
    /// Peer clusters in the federation.
    pub clusters: Vec<ClusterPeer>,
}

/// A peer cluster in a federation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPeer {
    /// Name of the peer cluster.
    pub name: String,
    /// gRPC address of the peer controller (e.g., "http://peer-ctrl:6817").
    pub address: String,
}

/// Device discovery and CDI configuration.
///
/// `auto_detect` is AMD KFD sysfs only; other vendors need on-disk CDI specs or GRES
/// `file` entries.
///
/// ```toml
/// [devices]
/// auto_detect = true
/// cdi_spec_dirs = ["/opt/vendor/cdi"]
///
/// [[devices.gres]]
/// name = "gpu"
/// file = "/dev/dri/renderD[128-129]"
/// flags = ["amd_gpu_env"]
///
/// [[devices.gres]]
/// name = "bandwidth"
/// type = "lustre"
/// count = 4096
/// flags = ["count_only"]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicesConfig {
    /// Discover GPUs from KFD when the CDI cache is empty. AMD only for now.
    #[serde(default = "default_true_fn")]
    pub auto_detect: bool,

    /// Additional directories to scan for CDI spec files, beyond the
    /// defaults (`/etc/cdi`, `/var/run/cdi`).
    #[serde(default)]
    pub cdi_spec_dirs: Vec<String>,

    /// File-based or countable GRES pools alongside CDI-discovered devices.
    #[serde(default)]
    pub gres: Vec<DevicesGresEntry>,
}

impl Default for DevicesConfig {
    fn default() -> Self {
        Self {
            auto_detect: true,
            cdi_spec_dirs: Vec::new(),
            gres: Vec::new(),
        }
    }
}

/// A `[[devices.gres]]` entry (Slurm-compatible syntax).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicesGresEntry {
    pub name: String,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub multiple_files: Option<String>,
    #[serde(default)]
    pub count: Option<u64>,
    #[serde(default)]
    pub cores: Option<String>,
    #[serde(default)]
    pub links: Option<String>,
    #[serde(default)]
    pub flags: Vec<String>,
}

/// Node admission control configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionConfig {
    #[serde(default)]
    pub mode: AdmissionMode,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            mode: AdmissionMode::Open,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AdmissionMode {
    #[default]
    Open,
    Token,
}

/// PMIx plugin settings for `--mpi=pmix` jobs (batch launch and srun steps).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MpiConfig {
    #[serde(default = "default_mpi_plugin_dir")]
    pub plugin_dir: String,
    #[serde(default)]
    pub pmix_plugin: String,
    #[serde(default = "default_pmix_tmpdir")]
    pub pmix_tmpdir: String,
    #[serde(default = "default_pmix_min_version")]
    pub pmix_min_version: String,
    #[serde(default = "default_modex_connect_timeout_secs")]
    pub modex_connect_timeout_secs: u32,
    #[serde(default = "default_modex_fence_timeout_secs")]
    pub modex_fence_timeout_secs: u32,
    #[serde(default = "default_modex_verify_timeout_secs")]
    pub modex_verify_timeout_secs: u32,
}

fn default_mpi_plugin_dir() -> String {
    "/usr/lib/spur".into()
}

fn default_pmix_tmpdir() -> String {
    "/tmp/spur-pmix".into()
}

fn default_pmix_min_version() -> String {
    "4.1.0".into()
}

fn default_modex_connect_timeout_secs() -> u32 {
    5
}

fn default_modex_fence_timeout_secs() -> u32 {
    120
}

fn default_modex_verify_timeout_secs() -> u32 {
    30
}

impl Default for MpiConfig {
    fn default() -> Self {
        Self {
            plugin_dir: default_mpi_plugin_dir(),
            pmix_plugin: String::new(),
            pmix_tmpdir: default_pmix_tmpdir(),
            pmix_min_version: default_pmix_min_version(),
            modex_connect_timeout_secs: default_modex_connect_timeout_secs(),
            modex_fence_timeout_secs: default_modex_fence_timeout_secs(),
            modex_verify_timeout_secs: default_modex_verify_timeout_secs(),
        }
    }
}

impl MpiConfig {
    pub fn resolve_pmix_plugin_path(&self) -> std::path::PathBuf {
        if !self.pmix_plugin.is_empty() {
            return std::path::PathBuf::from(&self.pmix_plugin);
        }
        std::path::Path::new(&self.plugin_dir).join("spur_mpi_pmix.so")
    }
}

/// Resolved limits an agent applies to every job it launches, as opposed to
/// the per-job sizes that come from the allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobLimits {
    /// Applied before exec, while still privileged.
    pub memlock: MemlockLimit,
}

impl Default for JobLimits {
    fn default() -> Self {
        Self {
            memlock: MemlockLimit::Unlimited,
        }
    }
}

/// POSIX per-process resource limits (`RLIMIT_*`) applied to job steps at launch.
///
/// Distinct from QoS/association caps (scheduling policy) and cgroup limits
/// (derived per-job from the allocation). This is where Tier 2 propagation
/// controls (`propagate`, `propagate_except`) will live.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlimitsConfig {
    /// RLIMIT_MEMLOCK applied to job processes.
    /// "unlimited" (default) | "inherit" | byte count as string.
    #[serde(default = "default_memlock")]
    pub memlock: String,
}

fn default_memlock() -> String {
    "unlimited".into()
}

impl Default for RlimitsConfig {
    fn default() -> Self {
        Self {
            memlock: default_memlock(),
        }
    }
}

/// Parsed value for RLIMIT_MEMLOCK configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemlockLimit {
    /// Set both soft and hard to RLIM_INFINITY.
    Unlimited,
    /// Do not call setrlimit; inherit from spurd's process.
    Inherit,
    /// Set both soft and hard to a fixed byte count.
    Bytes(u64),
}

impl RlimitsConfig {
    pub fn memlock_limit(&self) -> Result<MemlockLimit, ConfigError> {
        match self.memlock.trim().to_lowercase().as_str() {
            "unlimited" | "" => Ok(MemlockLimit::Unlimited),
            "inherit" => Ok(MemlockLimit::Inherit),
            other => match other.parse::<u64>() {
                Ok(0) => Ok(MemlockLimit::Unlimited),
                Ok(n) => Ok(MemlockLimit::Bytes(n)),
                Err(_) => Err(ConfigError::InvalidValue {
                    field: "rlimits.memlock".into(),
                    value: format!(
                        "{:?} (expected \"unlimited\", \"inherit\", or byte count)",
                        other
                    ),
                }),
            },
        }
    }
}

/// CFS period that cgroup-v2 CPU quotas are expressed against.
const CPU_PERIOD_US: u64 = 100_000;

/// cgroup-v2 enforcement for native-host jobs — a focused `cgroup.conf` without
/// the plugin selectors. Slurm defaults permissive here, Spur enforcing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CgroupConfig {
    /// Master switch. When false, spurd creates no cgroup and applies no limits.
    #[serde(default = "default_true_fn")]
    pub enabled: bool,
    /// Refuse to launch when a constraint cannot be applied, rather than warning and
    /// running unconstrained. Off by default so a bad host degrades, not stalls.
    #[serde(default)]
    pub required: bool,
    /// Pin the job to its allocated cores via `cpuset.cpus` (`ConstrainCores`).
    #[serde(default = "default_true_fn")]
    pub constrain_cores: bool,
    /// Also cap CPU time with a CFS quota (`cpu.max`). Off by default: Slurm bounds
    /// whole cores with the cpuset alone, and a quota on top only risks throttling.
    #[serde(default)]
    pub cpu_quota: bool,
    /// Cap resident memory via `memory.max` + `memory.high`
    /// (`ConstrainRAMSpace`).
    #[serde(default = "default_true_fn")]
    pub constrain_ram_space: bool,
    /// Hard ceiling as a percent of the job's budget (`AllowedRAMSpace`).
    /// `memory.high` stays at 100%, so above 100 reclaims first and kills later.
    #[serde(default = "default_allowed_ram_percent")]
    pub allowed_ram_percent: u32,
    /// Bound swap via `memory.swap.max` (`ConstrainSwapSpace`). Off by default,
    /// as in Slurm: turning it on kills jobs that used to survive by swapping.
    #[serde(default)]
    pub constrain_swap: bool,
    /// Swap allowance as a percent of the memory budget (`AllowedSwapSpace`), 0 =
    /// none. Uncapped above 100, as in Slurm: swap may legitimately exceed RAM.
    #[serde(default)]
    pub allowed_swap_percent: u32,
    /// Floor for `memory.max`, in MiB (`MinRAMSpace`). Below this a job is
    /// killed during its own startup, before anything can clean up after it.
    #[serde(default = "default_min_ram_mb")]
    pub min_ram_mb: u64,
    /// Kill the whole job on OOM (`memory.oom.group`), not one process. Slurm's
    /// `OOMKillStep` defaults off; on here, since partial kills fail murkily.
    #[serde(default = "default_true_fn")]
    pub oom_kill_job: bool,
    /// Restrict the job to its allocated device nodes (`ConstrainDevices`) via a
    /// cgroup-v2 BPF device filter. Default-deny: no allocation, pseudo-devices only.
    #[serde(default = "default_true_fn")]
    pub constrain_devices: bool,
    /// Extra device paths every job may open, on top of its allocation and the
    /// host-infrastructure list: the escape hatch for a site device the defaults miss.
    #[serde(default)]
    pub extra_device_paths: Vec<String>,
}

fn default_min_ram_mb() -> u64 {
    30
}

fn default_allowed_ram_percent() -> u32 {
    100
}

impl Default for CgroupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            required: false,
            constrain_cores: true,
            cpu_quota: false,
            constrain_ram_space: true,
            allowed_ram_percent: default_allowed_ram_percent(),
            constrain_swap: false,
            allowed_swap_percent: 0,
            min_ram_mb: default_min_ram_mb(),
            oom_kill_job: true,
            constrain_devices: true,
            extra_device_paths: Vec::new(),
        }
    }
}

/// Node health-check configuration.
///
/// The controller submits each configured check as a real scheduled job that
/// requests its nodes exclusively, so it runs only when a node is idle (the
/// scheduler enforces the "probe only when idle" property — no special-case
/// code). A non-zero exit drains the node with the check's output as the
/// reason; a pass auto-resumes a node the health system had drained. Health-
/// check jobs are visible in `squeue`/RPC/audit and run inside a cgroup like any
/// other job. Empty `checks` = no health checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// The configured checks (`[[health.checks]]`).
    #[serde(default)]
    pub checks: Vec<HealthCheck>,
    /// Cap on how many nodes may be in health-check activity (a pending or
    /// running `_spur-health.*` job) at once, so a restart, failover, or first
    /// enable can't set off a fleet-wide check — and, downstream, a fleet-wide
    /// force-drain — wave. An absolute count (`"5"`) or a percentage of eligible
    /// nodes (`"10%"`, rounded up, at least 1); `"0"` or `"0%"` disables the cap.
    #[serde(default = "default_health_max_unavailable")]
    pub max_unavailable: String,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            checks: Vec::new(),
            max_unavailable: default_health_max_unavailable(),
        }
    }
}

/// One health check, run as a scheduled job on every eligible node on an
/// interval. `nodes = 2` (with topology) expresses a pairwise fabric check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Fully-qualified path to the check program; run as the job's command.
    pub program: String,
    /// Nodes the check job requests exclusively (1 = per-node).
    #[serde(default = "default_health_nodes")]
    pub nodes: u32,
    /// Re-run the check on each eligible node roughly this often.
    #[serde(default = "default_health_interval")]
    pub interval_secs: u64,
    /// Kill the check job and treat it as failed after this long.
    #[serde(default = "default_health_timeout")]
    pub timeout_secs: u64,
    /// If the check job cannot start within this long because the node stays
    /// busy, soft-drain the node so it empties and the check can run. `0` waits
    /// indefinitely for the node to go idle (never forces a drain).
    #[serde(default = "default_health_max_wait")]
    pub max_wait_secs: u64,
    /// Unix user name the check runs as (for display and accounting). Defaults
    /// to an unprivileged user. A privileged probe needs `user`/`uid`/`gid` set
    /// to root (`0`) *and* `allow_root_jobs` enabled on the agents.
    #[serde(default = "default_health_user")]
    pub user: String,
    /// Unix uid the check runs as. Defaults to `nobody` (65534). The agent must
    /// be able to run jobs as this uid (a non-root spurd can only run as its own
    /// uid, so set this to that uid on such a node).
    #[serde(default = "default_health_uid")]
    pub uid: u32,
    /// Unix gid the check runs as. Defaults to `nogroup` (65534).
    #[serde(default = "default_health_gid")]
    pub gid: u32,
}

fn default_health_nodes() -> u32 {
    1
}

fn default_health_user() -> String {
    "nobody".to_string()
}

fn default_health_uid() -> u32 {
    65534
}

fn default_health_gid() -> u32 {
    65534
}

fn default_health_interval() -> u64 {
    300
}

fn default_health_timeout() -> u64 {
    60
}

fn default_health_max_wait() -> u64 {
    600
}

fn default_health_max_unavailable() -> String {
    "10%".to_string()
}

/// Parse a `max_unavailable` value ("5" or "10%") against the eligible-node
/// count into an absolute cap, or `None` if it is malformed. `0`/`0%` = no cap
/// (returns `Some(0)`). A positive percentage rounds up and is at least 1, so a
/// check can always make progress on a small fleet.
fn parse_health_cap(value: &str, eligible_nodes: usize) -> Option<usize> {
    let value = value.trim();
    if let Some(pct) = value.strip_suffix('%') {
        let pct: usize = pct.trim().parse().ok()?;
        if pct == 0 {
            return Some(0);
        }
        Some((eligible_nodes * pct).div_ceil(100).max(1))
    } else {
        value.parse().ok()
    }
}

impl HealthConfig {
    /// Whether any health checking is configured.
    pub fn is_enabled(&self) -> bool {
        !self.checks.is_empty()
    }

    /// Resolve `max_unavailable` to an absolute node cap given the eligible-node
    /// count. `0` = unlimited. Falls back to unlimited on a malformed value,
    /// which `validate()` rejects at config load, so this is never reached.
    pub fn concurrency_cap(&self, eligible_nodes: usize) -> usize {
        parse_health_cap(&self.max_unavailable, eligible_nodes).unwrap_or(0)
    }
}

/// Resolved cgroup-v2 control-file values for one job. `None`/empty means
/// "leave the kernel default", which is not the same as a limit of zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CgroupLimits {
    pub cpu_quota_us: Option<u64>,
    pub cpu_period_us: u64,
    pub memory_max_bytes: Option<u64>,
    pub memory_high_bytes: Option<u64>,
    pub swap_max_bytes: Option<u64>,
    pub oom_kill_job: bool,
    pub pids_max: u64,
    pub cpuset_cpus: Vec<u32>,
}

impl CgroupConfig {
    /// `(memory.max, memory.high, memory.swap.max)` as Slurm's cgroup/v2 plugin
    /// computes them. Swap-without-RAM: the RAM ceiling becomes the RAM+swap total.
    fn memory_limits(&self, memory_mb: u64) -> (Option<u64>, Option<u64>, Option<u64>) {
        // A job with no memory budget gets no ceiling at all. Bounding swap on
        // its own here would forbid swap while RAM stayed unlimited.
        if memory_mb == 0 || !(self.constrain_ram_space || self.constrain_swap) {
            return (None, None, None);
        }
        let memory_bytes = memory_mb.saturating_mul(1024 * 1024);
        let min_ram_bytes = self.min_ram_mb.saturating_mul(1024 * 1024);

        // The percentage only exists to build the ceilings from, so it reverts
        // to 100% when RAM itself is unconstrained.
        let ram_percent = if self.constrain_ram_space {
            self.allowed_ram_percent as u64
        } else {
            100
        };

        // Hard is AllowedRAMSpace% of the budget, soft 100%, both floored at
        // MinRAMSpace. A soft limit above the hard one is moot, so it collapses.
        let hard = (memory_bytes.saturating_mul(ram_percent) / 100).max(min_ram_bytes);
        let soft = memory_bytes.max(min_ram_bytes).min(hard);
        let combined = hard
            .saturating_add(memory_bytes.saturating_mul(self.allowed_swap_percent as u64) / 100);

        let memory_max = if self.constrain_ram_space {
            hard
        } else {
            combined
        };
        let swap_max = if self.constrain_swap {
            Some(combined.saturating_sub(memory_max))
        } else {
            None
        };
        (Some(memory_max), Some(soft), swap_max)
    }

    /// Limits for a job budgeted `cpus`/`memory_mb` on `cpu_ids`. `None` when
    /// disabled — the only place `enabled` is read, so it cannot be half-applied.
    pub fn limits_for(&self, cpus: u32, memory_mb: u64, cpu_ids: &[u32]) -> Option<CgroupLimits> {
        if !self.enabled {
            return None;
        }
        let cpus = cpus.max(1);
        let (memory_max_bytes, memory_high_bytes, swap_max_bytes) = self.memory_limits(memory_mb);
        let cpu_quota_us = if self.cpu_quota {
            Some(cpus as u64 * CPU_PERIOD_US)
        } else {
            None
        };

        Some(CgroupLimits {
            cpu_quota_us,
            cpu_period_us: CPU_PERIOD_US,
            memory_max_bytes,
            memory_high_bytes,
            swap_max_bytes,
            oom_kill_job: self.oom_kill_job,
            pids_max: (cpus as u64 * 256).max(1024),
            cpuset_cpus: if self.constrain_cores {
                cpu_ids.to_vec()
            } else {
                Vec::new()
            },
        })
    }
}

impl SlurmConfig {
    /// Load from a TOML file.
    pub fn load_from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load from a TOML string.
    pub fn load_from_str(s: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(s)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.cluster_name.is_empty() {
            return Err(ConfigError::MissingField("cluster_name".into()));
        }
        if self.controller.max_batch_requeue == 0 {
            return Err(ConfigError::InvalidValue {
                field: "controller.max_batch_requeue".into(),
                value: "0 (must be at least 1)".into(),
            });
        }
        // Zero would clamp every launch-failure hold to zero, restoring the tight
        // re-dispatch loop the backoff exists to prevent. The upper bound keeps
        // the hold instant representable.
        if self.controller.max_launch_backoff_secs == 0
            || self.controller.max_launch_backoff_secs > MAX_LAUNCH_BACKOFF_SECS
        {
            return Err(ConfigError::InvalidValue {
                field: "controller.max_launch_backoff_secs".into(),
                value: format!(
                    "{} (must be between 1 and {})",
                    self.controller.max_launch_backoff_secs, MAX_LAUNCH_BACKOFF_SECS
                ),
            });
        }
        // Guardrail: retention beyond the cap defeats the memory bound.
        if self.controller.terminal_job_retention_secs > MAX_TERMINAL_JOB_RETENTION_SECS {
            return Err(ConfigError::InvalidValue {
                field: "controller.terminal_job_retention_secs".into(),
                value: format!(
                    "{} (must be at most {})",
                    self.controller.terminal_job_retention_secs, MAX_TERMINAL_JOB_RETENTION_SECS
                ),
            });
        }
        // Capped at the same 1-day ceiling as max_launch_backoff_secs — a longer
        // dispatch cooldown has no legitimate operational use.
        if self.controller.dispatch_reject_cooldown_secs > MAX_LAUNCH_BACKOFF_SECS {
            return Err(ConfigError::InvalidValue {
                field: "controller.dispatch_reject_cooldown_secs".into(),
                value: format!(
                    "{} (must be at most {})",
                    self.controller.dispatch_reject_cooldown_secs, MAX_LAUNCH_BACKOFF_SECS
                ),
            });
        }
        // Keepalive detection is interval + timeout, so an hour-long interval silently disables
        // the liveness this exists to provide.
        for (field, value) in [
            (
                "controller.agent_connect_timeout_secs",
                self.controller.agent_connect_timeout_secs,
            ),
            (
                "controller.agent_keepalive_interval_secs",
                self.controller.agent_keepalive_interval_secs,
            ),
            (
                "controller.agent_keepalive_timeout_secs",
                self.controller.agent_keepalive_timeout_secs,
            ),
        ] {
            if value > MAX_AGENT_CHANNEL_TIMEOUT_SECS {
                return Err(ConfigError::InvalidValue {
                    field: field.into(),
                    value: format!("{value} (must be at most {MAX_AGENT_CHANNEL_TIMEOUT_SECS})"),
                });
            }
        }
        // Zero is "already overdue" to hyper, so it would tear down every channel that goes quiet
        // — the opposite of the liveness keepalive exists to provide.
        if self.controller.agent_keepalive_interval_secs > 0
            && self.controller.agent_keepalive_timeout_secs == 0
        {
            return Err(ConfigError::InvalidValue {
                field: "controller.agent_keepalive_timeout_secs".into(),
                value: "0 (must be greater than 0 unless agent_keepalive_interval_secs is 0)"
                    .into(),
            });
        }
        // A timed-out node is cooled down for this span, so it feeds the same map and needs the
        // same ceiling.
        if self.controller.dispatch_timeout_secs > MAX_LAUNCH_BACKOFF_SECS {
            return Err(ConfigError::InvalidValue {
                field: "controller.dispatch_timeout_secs".into(),
                value: format!(
                    "{} (must be at most {})",
                    self.controller.dispatch_timeout_secs, MAX_LAUNCH_BACKOFF_SECS
                ),
            });
        }
        // Zero collapses the consumption window, so every QOS reads zero consumed
        // and every GrpWall budget silently stops applying while `sacctmgr show
        // qos` still displays the cap — the exact failure the limit exists to
        // prevent, one layer up.
        if self.accounting.grp_wall_window_days == 0
            || self.accounting.grp_wall_window_days > MAX_GRP_WALL_WINDOW_DAYS
        {
            return Err(ConfigError::InvalidValue {
                field: "accounting.grp_wall_window_days".into(),
                value: format!(
                    "{} (must be between 1 and {})",
                    self.accounting.grp_wall_window_days, MAX_GRP_WALL_WINDOW_DAYS
                ),
            });
        }
        // Zero collapses every ceiling onto `min_ram_mb` — a typo, never an intent.
        // Above 100 is legal (as in Slurm), so only zero is rejected.
        if self.cgroup.allowed_ram_percent == 0 {
            return Err(ConfigError::InvalidValue {
                field: "cgroup.allowed_ram_percent".into(),
                value: "0 (must be at least 1)".into(),
            });
        }
        // A limit below the client's ping interval would reap a live client
        // between its own keepalives. Require at least two intervals of slack.
        let inactive = self.scheduler.inactive_limit_secs;
        if inactive > 0 && u64::from(inactive) < 2 * KEEPALIVE_INTERVAL_SECS {
            return Err(ConfigError::InvalidValue {
                field: "scheduler.inactive_limit_secs".into(),
                value: format!(
                    "{inactive} (0 disables reaping; otherwise must be >= {})",
                    2 * KEEPALIVE_INTERVAL_SECS
                ),
            });
        }
        // `plugin` used to be parsed and never read, so an operator could set "munge" (or a typo)
        // and get no authentication with no warning — worse than the field not existing. Reject
        // anything unimplemented instead of silently ignoring it.
        match self.auth.plugin.as_str() {
            "jwt" | "none" => {}
            "munge" => {
                return Err(ConfigError::InvalidValue {
                    field: "auth.plugin".into(),
                    value: "munge (not implemented; use \"jwt\", or \"none\" to disable)".into(),
                })
            }
            other => {
                return Err(ConfigError::InvalidValue {
                    field: "auth.plugin".into(),
                    value: format!("{other} (expected \"jwt\" or \"none\")"),
                })
            }
        }
        // "none" and a mode that promises enforcement are contradictory; fail rather than pick one.
        if self.auth.plugin == "none" && self.auth.mode == AuthMode::Required {
            return Err(ConfigError::InvalidValue {
                field: "auth.mode".into(),
                value: "required with auth.plugin = \"none\" (no credential can be verified)"
                    .into(),
            });
        }
        // Without a key, `required` would fall back to a well-known signing constant —
        // forgeable, so strictly worse than the default. Refuse to start instead.
        let resolved_jwt_key = self.auth.resolved_jwt_key()?;
        if self.auth.mode == AuthMode::Required
            && resolved_jwt_key.as_deref().unwrap_or("").is_empty()
        {
            return Err(ConfigError::InvalidValue {
                field: "auth.jwt_key".into(),
                value:
                    "unset with auth.mode = \"required\" (set a secret signing key; the hardened \
                        mode cannot verify credentials without one)"
                        .into(),
            });
        }

        if self.cluster.enabled {
            if self.cluster.distro != "k0s" {
                return Err(ConfigError::InvalidValue {
                    field: "cluster.distro".into(),
                    value: format!("{} (only \"k0s\" is supported)", self.cluster.distro),
                });
            }
            if let Err(msg) =
                crate::k0s::validate_control_plane_replicas(self.cluster.control_plane_replicas)
            {
                return Err(ConfigError::InvalidValue {
                    field: "cluster.control_plane_replicas".into(),
                    value: msg,
                });
            }
            if self.cluster.k8s_provisioning_timeout_secs == 0 {
                return Err(ConfigError::InvalidValue {
                    field: "cluster.k8s_provisioning_timeout_secs".into(),
                    value: "0 (must be > 0; a cluster would degrade before any node could start)"
                        .into(),
                });
            }
            // The mesh CIDR feeds the k0s IPAM (AddressPool) exactly like pod/service, so assert it is
            // a valid IPv4 CIDR in the same pass — otherwise a malformed wg_cidr bypasses validate()
            // and fails every reconcile tick in AddressPool::new, leaving the cluster silently stuck.
            for (field, cidr) in [
                ("network.wg_cidr", &self.network.wg_cidr),
                ("cluster.pod_cidr", &self.cluster.pod_cidr),
                ("cluster.service_cidr", &self.cluster.service_cidr),
            ] {
                if !is_valid_cidr(cidr) {
                    return Err(ConfigError::InvalidValue {
                        field: field.into(),
                        value: cidr.clone(),
                    });
                }
            }
            // Pods are carved into per-node /24s (`carve_pod_cidr`), so the pod CIDR must be <= /24.
            // is_valid_cidr only bounds the prefix at /32, so this would otherwise pass here and then
            // fail every reconcile tick with only a warn! — the same silent-stuck state.
            if cidr_prefix(&self.cluster.pod_cidr) > 24 {
                return Err(ConfigError::InvalidValue {
                    field: "cluster.pod_cidr".into(),
                    value: format!(
                        "{} (prefix must be <= /24 to carve per-node /24s)",
                        self.cluster.pod_cidr
                    ),
                });
            }
            // Mesh, pod, and service ranges must be mutually non-overlapping.
            for (fa, a, fb, b) in [
                (
                    "cluster.pod_cidr",
                    &self.cluster.pod_cidr,
                    "cluster.service_cidr",
                    &self.cluster.service_cidr,
                ),
                (
                    "network.wg_cidr",
                    &self.network.wg_cidr,
                    "cluster.pod_cidr",
                    &self.cluster.pod_cidr,
                ),
                (
                    "network.wg_cidr",
                    &self.network.wg_cidr,
                    "cluster.service_cidr",
                    &self.cluster.service_cidr,
                ),
            ] {
                if cidrs_overlap(a, b) {
                    return Err(ConfigError::InvalidValue {
                        field: format!("{fa}/{fb}"),
                        value: format!("{a} overlaps {b}"),
                    });
                }
            }
            if !matches!(
                self.cluster.storage_provisioner.as_str(),
                "local-path" | "none"
            ) {
                return Err(ConfigError::InvalidValue {
                    field: "cluster.storage_provisioner".into(),
                    value: format!(
                        "{} (expected \"local-path\" or \"none\")",
                        self.cluster.storage_provisioner
                    ),
                });
            }
            // local_path_dir is interpolated verbatim into a JSON string in the generated manifest,
            // so an absolute path free of quotes/backslashes/whitespace/control chars keeps the JSON
            // valid and can't inject. Only meaningful when local-path is the chosen provisioner.
            if self.cluster.storage_provisioner == "local-path" {
                let d = &self.cluster.local_path_dir;
                if !d.starts_with('/')
                    || d.chars()
                        .any(|c| c == '"' || c == '\\' || c.is_whitespace() || c.is_control())
                {
                    return Err(ConfigError::InvalidValue {
                        field: "cluster.local_path_dir".into(),
                        value: format!(
                            "{d} (must be an absolute path with no quotes, backslashes, or whitespace)"
                        ),
                    });
                }
            }
        }

        // Reject malformed partition time strings at load rather than silently
        // treating them as UNLIMITED (a typo like "1 hour" must fail, not
        // remove the cap).
        for pc in &self.partitions {
            let max = pc
                .max_time
                .as_ref()
                .map(|t| parse_partition_time(&format!("partitions.{}.max_time", pc.name), t))
                .transpose()?
                .flatten();
            let default = pc
                .default_time
                .as_ref()
                .map(|t| parse_partition_time(&format!("partitions.{}.default_time", pc.name), t))
                .transpose()?
                .flatten();
            // A finite DefaultTime above the partition's finite MaxTime would hand
            // a -t-less job a limit its own partition rejects, so it pends forever.
            if let (Some(max), Some(default)) = (max, default) {
                if default > max {
                    return Err(ConfigError::InvalidValue {
                        field: format!("partitions.{}.default_time", pc.name),
                        value: format!("{default}m exceeds max_time {max}m"),
                    });
                }
            }
        }

        // The health-check program is a path resolved on the node, so the
        // controller can't stat it — but an absolute path is required, so a
        // relative typo is caught here rather than draining every node when the
        // check fails to launch. Multi-node checks need topology-aware pairing
        // that does not exist yet, so reject them at load rather than skip and
        // log-spam every scheduler tick.
        for (i, check) in self.health.checks.iter().enumerate() {
            if !Path::new(&check.program).is_absolute() {
                return Err(ConfigError::InvalidValue {
                    field: format!("health.checks[{i}].program"),
                    value: format!("{:?} (must be an absolute path)", check.program),
                });
            }
            if check.nodes != 1 {
                return Err(ConfigError::InvalidValue {
                    field: format!("health.checks[{i}].nodes"),
                    value: format!(
                        "{} (only single-node checks are supported for now)",
                        check.nodes
                    ),
                });
            }
        }
        if parse_health_cap(&self.health.max_unavailable, 1).is_none() {
            return Err(ConfigError::InvalidValue {
                field: "health.max_unavailable".into(),
                value: format!(
                    "{:?} (expected a count like \"5\" or a percentage like \"10%\")",
                    self.health.max_unavailable
                ),
            });
        }
        Ok(())
    }

    /// Convert partition configs to Partition structs. Time strings are assumed
    /// valid here; [`Self::validate`] rejects malformed ones at load, so a stray
    /// `None` below only ever means UNLIMITED, never a swallowed parse error.
    pub fn build_partitions(&self) -> Vec<Partition> {
        self.partitions
            .iter()
            .map(|pc| Partition {
                name: pc.name.clone(),
                state: match pc.state.to_uppercase().as_str() {
                    "UP" => PartitionState::Up,
                    "DOWN" => PartitionState::Down,
                    "DRAIN" => PartitionState::Drain,
                    _ => PartitionState::Inactive,
                },
                is_default: pc.default,
                nodes: pc.nodes.clone(),
                selector: pc.selector.clone(),
                max_time_minutes: pc.max_time.as_ref().and_then(|t| parse_time_minutes(t)),
                default_time_minutes: pc.default_time.as_ref().and_then(|t| parse_time_minutes(t)),
                max_nodes: pc.max_nodes,
                min_nodes: pc.min_nodes,
                allow_accounts: pc.allow_accounts.clone(),
                allow_groups: pc.allow_groups.clone(),
                deny_accounts: pc.deny_accounts.clone(),
                deny_qos: pc.deny_qos.clone(),
                allow_qos: pc.allow_qos.clone(),
                priority_tier: pc.priority_tier,
                preempt_mode: match pc.preempt_mode.to_lowercase().as_str() {
                    "cancel" => PreemptMode::Cancel,
                    "requeue" => PreemptMode::Requeue,
                    "suspend" => PreemptMode::Suspend,
                    _ => PreemptMode::Off,
                },
                preempt_exempt_time: pc.preempt_exempt_time,
                ..Default::default()
            })
            .collect()
    }
}

/// Parse a partition time field, distinguishing UNLIMITED from a parse error:
/// `INFINITE`/`UNLIMITED` -> `Ok(None)`, a valid duration -> `Ok(Some(minutes))`,
/// anything else -> `Err`. Unlike [`parse_time_minutes`], a typo like `"1 hour"`
/// fails loudly instead of collapsing to UNLIMITED.
pub fn parse_partition_time(field: &str, s: &str) -> Result<Option<u32>, ConfigError> {
    let trimmed = s.trim();
    if trimmed.eq_ignore_ascii_case("INFINITE") || trimmed.eq_ignore_ascii_case("UNLIMITED") {
        return Ok(None);
    }
    match parse_time_minutes(trimmed) {
        Some(minutes) => Ok(Some(minutes)),
        None => Err(ConfigError::InvalidValue {
            field: field.into(),
            value: s.into(),
        }),
    }
}

/// Parse a time string to minutes. Accepts Slurm's documented `--time` grammar
/// (`minutes`, `minutes:seconds`, `hours:minutes:seconds`, `days-hours`,
/// `days-hours:minutes`, `days-hours:minutes:seconds`, `INFINITE`/`UNLIMITED`)
/// or a suffixed duration (`90m`, `1h`, `2d12h`, `30s`). Sub-minute remainders
/// round up.
pub fn parse_time_minutes(s: &str) -> Option<u32> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("INFINITE") || s.eq_ignore_ascii_case("UNLIMITED") {
        return None; // No limit
    }
    parse_slurm_time_minutes(s).or_else(|| {
        let secs = parse_suffix_duration_seconds(s)?;
        u32::try_from(secs.div_ceil(60)).ok()
    })
}

fn parse_slurm_time_minutes(s: &str) -> Option<u32> {
    // days-hours:minutes:seconds
    if let Some((days, rest)) = s.split_once('-') {
        let days: u32 = days.parse().ok()?;
        let hms = parse_hms(rest)?;
        return days.checked_mul(24 * 60)?.checked_add(hms);
    }

    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        1 => parts[0].parse().ok(),
        2 => {
            // MM:SS — Slurm's two-field form is minutes:seconds. Sum first, then
            // round up, so an unnormalised seconds field like "1:90" carries into
            // whole minutes rather than adding a flat one.
            let m: u64 = parts[0].parse().ok()?;
            let sec: u64 = parts[1].parse().ok()?;
            let total = m.checked_mul(60)?.checked_add(sec)?;
            u32::try_from(total.div_ceil(60)).ok()
        }
        3 => parse_hms(s),
        _ => None,
    }
}

/// Parse a time string to total seconds (not minutes).
///
/// Same accepted formats as [`parse_time_minutes`] (Slurm grammar plus suffixed
/// durations), but with second granularity: "N" → N minutes, "MM:SS" →
/// minutes+seconds, "H:MM:SS" → exact, "90s" → 90 seconds.
pub fn parse_time_seconds(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("INFINITE") || s.eq_ignore_ascii_case("UNLIMITED") {
        return None;
    }
    parse_slurm_time_seconds(s).or_else(|| parse_suffix_duration_seconds(s))
}

fn parse_slurm_time_seconds(s: &str) -> Option<u64> {
    // days-hours:minutes:seconds
    if let Some((days, rest)) = s.split_once('-') {
        let days: u64 = days.parse().ok()?;
        return days
            .checked_mul(86400)?
            .checked_add(parse_hms_seconds(rest)?);
    }

    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        1 => {
            // Just minutes → convert to seconds
            let mins: u64 = parts[0].parse().ok()?;
            mins.checked_mul(60)
        }
        2 => {
            // MM:SS — Slurm's two-field form is minutes:seconds, not hours:minutes.
            let m: u64 = parts[0].parse().ok()?;
            let sec: u64 = parts[1].parse().ok()?;
            m.checked_mul(60)?.checked_add(sec)
        }
        3 => {
            // HH:MM:SS
            let h: u64 = parts[0].parse().ok()?;
            let m: u64 = parts[1].parse().ok()?;
            let sec: u64 = parts[2].parse().ok()?;
            h.checked_mul(3600)?
                .checked_add(m.checked_mul(60)?)?
                .checked_add(sec)
        }
        _ => None,
    }
}

/// Sum a suffixed duration like `90m`, `1h40m`, `2d12h`, or `30s` to seconds.
/// Units `d/h/m/s` (case-insensitive); a bare number or trailing digits fail
/// (the Slurm grammar owns bare numbers).
fn parse_suffix_duration_seconds(s: &str) -> Option<u64> {
    let mut total: u64 = 0;
    let mut num: Option<u64> = None;
    let mut saw_unit = false;
    for c in s.chars() {
        if let Some(digit) = c.to_digit(10) {
            num = Some(
                num.unwrap_or(0)
                    .checked_mul(10)?
                    .checked_add(digit as u64)?,
            );
            continue;
        }
        let unit_secs: u64 = match c.to_ascii_lowercase() {
            'd' => 86_400,
            'h' => 3_600,
            'm' => 60,
            's' => 1,
            _ => return None,
        };
        let value = num.take()?;
        total = total.checked_add(value.checked_mul(unit_secs)?)?;
        saw_unit = true;
    }
    // Trailing digits without a unit, or no unit at all, is not a duration.
    if num.is_some() || !saw_unit {
        return None;
    }
    Some(total)
}

/// Parse an hours-first duration to exact seconds: bare hours, `hours:minutes`,
/// or `hours:minutes:seconds`. Also used by the minute-granularity parser.
fn parse_hms_seconds(s: &str) -> Option<u64> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        1 => {
            let h: u64 = parts[0].parse().ok()?;
            h.checked_mul(3600)
        }
        2 => {
            let h: u64 = parts[0].parse().ok()?;
            let m: u64 = parts[1].parse().ok()?;
            h.checked_mul(3600)?.checked_add(m.checked_mul(60)?)
        }
        3 => {
            let h: u64 = parts[0].parse().ok()?;
            let m: u64 = parts[1].parse().ok()?;
            let sec: u64 = parts[2].parse().ok()?;
            h.checked_mul(3600)?
                .checked_add(m.checked_mul(60)?)?
                .checked_add(sec)
        }
        _ => None,
    }
}

/// Parse an hours-first duration to minutes, rounding up leftover seconds.
fn parse_hms(s: &str) -> Option<u32> {
    u32::try_from(parse_hms_seconds(s)?.div_ceil(60)).ok()
}

/// Format minutes as D-HH:MM:SS or HH:MM:SS.
pub fn format_time(total_minutes: Option<u32>) -> String {
    match total_minutes {
        None => "UNLIMITED".into(),
        Some(mins) => {
            let days = mins / (24 * 60);
            let hours = (mins % (24 * 60)) / 60;
            let minutes = mins % 60;
            if days > 0 {
                format!("{}-{:02}:{:02}:00", days, hours, minutes)
            } else {
                format!("{:02}:{:02}:00", hours, minutes)
            }
        }
    }
}

/// Format seconds as D-HH:MM:SS or HH:MM:SS. Unlike [`format_time`] this keeps
/// second precision, so a sub-minute request is not rounded when reported.
pub fn format_time_seconds(total_seconds: Option<i64>) -> String {
    match total_seconds {
        None => "UNLIMITED".into(),
        Some(secs) => {
            let secs = secs.max(0);
            let days = secs / 86_400;
            let hours = (secs % 86_400) / 3_600;
            let minutes = (secs % 3_600) / 60;
            let seconds = secs % 60;
            if days > 0 {
                format!("{days}-{hours:02}:{minutes:02}:{seconds:02}")
            } else {
                format!("{hours:02}:{minutes:02}:{seconds:02}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn auth_config_reads_an_explicit_jwt_key_file() {
        let mut key_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(key_file, "key-from-file").unwrap();
        let config = SlurmConfig::load_from_str(&format!(
            "cluster_name = \"test\"\n[auth]\nplugin = \"jwt\"\njwt_key_file = {:?}\n",
            key_file.path()
        ))
        .unwrap();

        assert_eq!(
            config.auth.resolved_jwt_key().unwrap().as_deref(),
            Some("key-from-file")
        );
    }

    #[test]
    fn auth_config_preserves_a_file_like_jwt_key_as_an_inline_secret() {
        let key_file = tempfile::NamedTempFile::new().unwrap();
        let config = SlurmConfig::load_from_str(&format!(
            "cluster_name = \"test\"\n[auth]\nplugin = \"jwt\"\njwt_key = {:?}\n",
            key_file.path()
        ))
        .unwrap();

        assert_eq!(
            config.auth.resolved_jwt_key().unwrap().as_deref(),
            key_file.path().to_str()
        );
    }

    #[test]
    fn auth_config_rejects_a_blank_inline_jwt_key() {
        let error = SlurmConfig::load_from_str(
            "cluster_name = \"test\"\n[auth]\nplugin = \"jwt\"\njwt_key = \" \"\n",
        )
        .unwrap_err();

        assert!(error.to_string().contains("auth.jwt_key"));
        assert!(!AuthConfig {
            jwt_key: Some(" \n".into()),
            ..Default::default()
        }
        .has_jwt_key());
    }

    #[test]
    fn auth_config_rejects_a_blank_jwt_key_file() {
        let mut key_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(key_file).unwrap();
        let error = SlurmConfig::load_from_str(&format!(
            "cluster_name = \"test\"\n[auth]\nplugin = \"jwt\"\njwt_key_file = {:?}\n",
            key_file.path()
        ))
        .unwrap_err();

        assert!(error.to_string().contains("auth.jwt_key_file"));
        assert!(!AuthConfig {
            jwt_key_file: Some(key_file.path().display().to_string()),
            ..Default::default()
        }
        .has_jwt_key());
    }

    #[test]
    fn auth_config_rejects_a_directory_as_a_jwt_key_file() {
        let directory = tempfile::tempdir().unwrap();
        let config = AuthConfig {
            jwt_key_file: Some(directory.path().display().to_string()),
            ..Default::default()
        };

        let error = config.resolved_jwt_key().unwrap_err();
        assert!(error.to_string().contains("not a regular file"));
    }

    #[test]
    fn auth_config_rejects_jwt_key_and_file_together() {
        let config = AuthConfig {
            jwt_key: Some("inline-secret".into()),
            jwt_key_file: Some("/etc/spur/jwt.key".into()),
            ..Default::default()
        };

        let error = config.resolved_jwt_key().unwrap_err();
        assert!(error.to_string().contains("cannot both be set"));
    }

    #[test]
    fn cgroup_config_defaults_match_intended_posture() {
        let c = CgroupConfig::default();
        assert!(c.enabled);
        assert!(c.constrain_cores && c.constrain_ram_space);
        assert!(c.oom_kill_job);
        // Degrade rather than stop accepting work on a misconfigured host.
        assert!(!c.required);
        // Slurm bounds whole-core allocations with the cpuset alone (D4).
        assert!(!c.cpu_quota);
        // Off as in Slurm: bounding swap kills jobs that used to ride it out.
        assert!(!c.constrain_swap);
        assert_eq!(c.allowed_swap_percent, 0);
        assert_eq!(c.allowed_ram_percent, 100);
        assert_eq!(c.min_ram_mb, 30);
    }

    #[test]
    fn memory_high_tracks_memory_max_at_the_default_percent() {
        // AllowedRAMSpace=100: Slurm clamps the soft limit to the hard one, so
        // both land on the allocation.
        let l = CgroupConfig::default()
            .limits_for(4, 1024, &[])
            .expect("enabled");
        assert_eq!(l.memory_max_bytes, Some(1024 * 1024 * 1024));
        assert_eq!(l.memory_high_bytes, l.memory_max_bytes);
    }

    #[test]
    fn headroom_splits_the_soft_and_hard_ceilings() {
        // AllowedRAMSpace=150: reclaim at the allocation, OOM kill at 1.5x.
        let c = CgroupConfig {
            allowed_ram_percent: 150,
            ..CgroupConfig::default()
        };
        let l = c.limits_for(4, 1024, &[]).expect("enabled");
        assert_eq!(l.memory_high_bytes, Some(1024 * 1024 * 1024));
        assert_eq!(l.memory_max_bytes, Some(1536 * 1024 * 1024));
    }

    #[test]
    fn shrinking_the_hard_ceiling_pulls_the_soft_one_down_with_it() {
        // AllowedRAMSpace=50: a soft limit above the hard one is moot, so Slurm
        // collapses them.
        let c = CgroupConfig {
            allowed_ram_percent: 50,
            ..CgroupConfig::default()
        };
        let l = c.limits_for(4, 1024, &[]).expect("enabled");
        assert_eq!(l.memory_max_bytes, Some(512 * 1024 * 1024));
        assert_eq!(l.memory_high_bytes, l.memory_max_bytes);
    }

    #[test]
    fn swap_ceiling_is_a_percent_of_allocated_memory() {
        // 0% => memory.swap.max = 0, i.e. Slurm's `memsw - mem` with
        // AllowedSwapSpace=0 and AllowedRAMSpace=100.
        let c = CgroupConfig {
            constrain_swap: true,
            ..CgroupConfig::default()
        };
        assert_eq!(c.limits_for(4, 1024, &[]).unwrap().swap_max_bytes, Some(0));

        let c50 = CgroupConfig {
            constrain_swap: true,
            allowed_swap_percent: 50,
            ..CgroupConfig::default()
        };
        assert_eq!(
            c50.limits_for(4, 1024, &[]).unwrap().swap_max_bytes,
            Some(512 * 1024 * 1024)
        );

        let off = CgroupConfig {
            constrain_swap: false,
            ..CgroupConfig::default()
        };
        assert_eq!(off.limits_for(4, 1024, &[]).unwrap().swap_max_bytes, None);
    }

    #[test]
    fn swap_ceiling_is_independent_of_the_ram_percent() {
        // Slurm's `memsw - mem` cancels the AllowedRAMSpace term.
        let c = CgroupConfig {
            constrain_swap: true,
            allowed_ram_percent: 150,
            allowed_swap_percent: 25,
            ..CgroupConfig::default()
        };
        assert_eq!(
            c.limits_for(4, 1024, &[]).unwrap().swap_max_bytes,
            Some(256 * 1024 * 1024)
        );
    }

    #[test]
    fn memory_ceilings_cover_the_constrain_matrix() {
        // 1 GiB budget with AllowedSwapSpace=25%, checked against what Slurm's
        // cgroup/v2 plugin writes for each Constrain{RAMSpace,SwapSpace} pair.
        const GIB: u64 = 1024 * 1024 * 1024;
        let cases = [
            // Neither constrained: the memory cgroup is left alone entirely.
            (false, false, None, None, None),
            // RAM only: ceiling at the budget, swap left to the kernel.
            (true, false, Some(GIB), Some(GIB), None),
            // Swap only: the RAM ceiling becomes the RAM+swap total so the sum
            // stays bounded, and swap itself is then pinned to zero.
            (false, true, Some(GIB + GIB / 4), Some(GIB), Some(0)),
            // Both: swap gets its own allowance on top of the RAM ceiling.
            (true, true, Some(GIB), Some(GIB), Some(GIB / 4)),
        ];

        for (constrain_ram, constrain_swap, want_max, want_high, want_swap) in cases {
            let c = CgroupConfig {
                constrain_ram_space: constrain_ram,
                constrain_swap,
                allowed_swap_percent: 25,
                ..CgroupConfig::default()
            };
            let l = c.limits_for(1, 1024, &[]).expect("enabled");
            assert_eq!(
                (l.memory_max_bytes, l.memory_high_bytes, l.swap_max_bytes),
                (want_max, want_high, want_swap),
                "constrain_ram_space={constrain_ram}, constrain_swap={constrain_swap}"
            );
        }
    }

    #[test]
    fn ram_percent_is_ignored_when_ram_is_not_constrained() {
        // AllowedRAMSpace only shapes the ceiling Spur derives the swap bound
        // from, so with RAM unconstrained Slurm reverts it to 100%.
        let c = CgroupConfig {
            constrain_ram_space: false,
            constrain_swap: true,
            allowed_ram_percent: 150,
            allowed_swap_percent: 50,
            ..CgroupConfig::default()
        };
        let l = c.limits_for(1, 1024, &[]).expect("enabled");
        assert_eq!(l.memory_max_bytes, Some(1536 * 1024 * 1024));
        assert_eq!(l.memory_high_bytes, Some(1024 * 1024 * 1024));
        assert_eq!(l.swap_max_bytes, Some(0));
    }

    #[test]
    fn unbounded_memory_leaves_swap_unbounded() {
        // Writing memory.swap.max = 0 for a job with no memory limit would
        // forbid swap while RAM stays unlimited — a pair Slurm never emits.
        let l = CgroupConfig::default()
            .limits_for(4, 0, &[])
            .expect("enabled");
        assert_eq!(l.memory_max_bytes, None);
        assert_eq!(l.memory_high_bytes, None);
        assert_eq!(l.swap_max_bytes, None);
    }

    #[test]
    fn memory_limits_floor_at_min_ram_mb() {
        let l = CgroupConfig::default()
            .limits_for(1, 8, &[])
            .expect("enabled");
        assert_eq!(l.memory_max_bytes, Some(30 * 1024 * 1024));
        assert_eq!(l.memory_high_bytes, Some(30 * 1024 * 1024));
    }

    #[test]
    fn cpu_quota_is_opt_in() {
        assert_eq!(
            CgroupConfig::default()
                .limits_for(4, 0, &[])
                .unwrap()
                .cpu_quota_us,
            None
        );
        let q = CgroupConfig {
            cpu_quota: true,
            ..CgroupConfig::default()
        };
        assert_eq!(q.limits_for(4, 0, &[]).unwrap().cpu_quota_us, Some(400_000));
    }

    #[test]
    fn core_pinning_can_be_turned_off() {
        let on = CgroupConfig::default()
            .limits_for(2, 0, &[0, 1])
            .expect("enabled");
        assert_eq!(on.cpuset_cpus, vec![0, 1]);

        let off = CgroupConfig {
            constrain_cores: false,
            ..CgroupConfig::default()
        };
        assert!(off
            .limits_for(2, 0, &[0, 1])
            .unwrap()
            .cpuset_cpus
            .is_empty());
    }

    #[test]
    fn disabled_yields_no_limits_at_all() {
        let off = CgroupConfig {
            enabled: false,
            ..CgroupConfig::default()
        };
        assert!(off.limits_for(4, 1024, &[0, 1]).is_none());
    }

    #[test]
    fn missing_cgroup_section_uses_defaults() {
        let cfg = SlurmConfig::load_from_str("cluster_name = \"test\"").expect("parses");
        assert!(cfg.cgroup.enabled);
        assert_eq!(cfg.cgroup.allowed_swap_percent, 0);
    }

    #[test]
    fn devices_are_constrained_unless_a_config_opts_out() {
        // A `spur.conf` deployed before the field existed must still load, and land
        // on isolation; the `Default` impl and the serde default must agree.
        assert!(CgroupConfig::default().constrain_devices);

        let upgraded =
            SlurmConfig::load_from_str("cluster_name = \"t\"\n[cgroup]\nconstrain_swap = true\n")
                .expect("parses");
        assert!(
            upgraded.cgroup.constrain_swap,
            "the field the config does set still applies"
        );
        assert!(upgraded.cgroup.constrain_devices);

        let opted_out = SlurmConfig::load_from_str(
            "cluster_name = \"t\"\n[cgroup]\nconstrain_devices = false\n",
        )
        .expect("parses");
        assert!(!opted_out.cgroup.constrain_devices);
    }

    #[test]
    fn extra_device_paths_default_to_none_and_parse_as_a_list() {
        assert!(CgroupConfig::default().extra_device_paths.is_empty());

        let without =
            SlurmConfig::load_from_str("cluster_name = \"t\"\n[cgroup]\nrequired = true\n")
                .expect("parses");
        assert!(
            without.cgroup.extra_device_paths.is_empty(),
            "a config written before the field existed must still load, granting nothing extra"
        );

        let with = SlurmConfig::load_from_str(
            "cluster_name = \"t\"\n[cgroup]\nextra_device_paths = [\"/dev/site-accel0\"]\n",
        )
        .expect("parses");
        assert_eq!(with.cgroup.extra_device_paths, ["/dev/site-accel0"]);
    }

    #[test]
    fn shipped_example_config_parses() {
        // Releases ship this as etc/spur.conf.example, so a stale field here
        // reaches operators as a config that will not load.
        let example = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/spur.conf"
        ));
        let cfg = SlurmConfig::load_from_str(example).expect("examples/spur.conf parses");
        assert!(cfg.cgroup.constrain_swap, "example should bound swap");
    }

    #[test]
    fn swap_percent_above_100_is_accepted() {
        // Slurm applies no upper bound to AllowedSwapSpace, so a `cgroup.conf`
        // carrying one must not stop the controller from starting.
        let cfg = SlurmConfig::load_from_str(
            "cluster_name = \"test\"\n[cgroup]\nconstrain_swap = true\nallowed_swap_percent = 200\n",
        )
        .expect("parses");
        assert_eq!(cfg.cgroup.allowed_swap_percent, 200);

        // 200% of a 1 GiB grant is 2 GiB of swap on top of the RAM ceiling.
        let l = cfg.cgroup.limits_for(1, 1024, &[]).expect("enabled");
        assert_eq!(l.memory_max_bytes, Some(1024 * 1024 * 1024));
        assert_eq!(l.swap_max_bytes, Some(2048 * 1024 * 1024));
    }

    #[test]
    fn zero_ram_percent_is_rejected_but_over_100_is_allowed() {
        let err = SlurmConfig::load_from_str(
            "cluster_name = \"test\"\n[cgroup]\nallowed_ram_percent = 0\n",
        )
        .expect_err("must not validate");
        assert!(matches!(err, ConfigError::InvalidValue { .. }));

        // Over-committing the node is a legitimate (documented) choice, as in Slurm.
        let cfg = SlurmConfig::load_from_str(
            "cluster_name = \"test\"\n[cgroup]\nallowed_ram_percent = 150\n",
        )
        .expect("parses");
        assert_eq!(cfg.cgroup.allowed_ram_percent, 150);
    }

    #[test]
    fn test_controller_endpoints_single_host() {
        let cfg = ControllerConfig::default();
        assert_eq!(cfg.endpoints(), vec!["http://localhost:6817"]);
    }

    #[test]
    fn cluster_config_defaults_are_disabled_and_sane() {
        let c = ClusterConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.distro, "k0s");
        assert_eq!(c.pod_cidr, "10.42.0.0/16");
        assert_eq!(c.service_cidr, "10.43.0.0/16");
        assert_eq!(c.cni_mtu, 1450);
        assert_eq!(c.cni, "kuberouter");
        assert_eq!(c.storage_provisioner, "local-path");
        assert_eq!(c.local_path_dir, "/var/lib/local-path-provisioner");
    }

    #[test]
    fn cluster_config_round_trips_from_toml() {
        let toml = r#"
cluster_name = "test"

[cluster]
enabled = true
pod_cidr = "10.60.0.0/16"
control_plane_node = "head-node"
cni = "calico"
cni_mtu = 1400
"#;
        let cfg = SlurmConfig::load_from_str(toml).expect("valid cluster config");
        assert!(cfg.cluster.enabled);
        assert_eq!(cfg.cluster.distro, "k0s"); // default fills in
        assert_eq!(cfg.cluster.pod_cidr, "10.60.0.0/16");
        assert_eq!(cfg.cluster.control_plane_node.as_deref(), Some("head-node"));
        assert_eq!(cfg.cluster.service_cidr, "10.43.0.0/16"); // default
        assert_eq!(cfg.cluster.cni, "calico");
        assert_eq!(cfg.cluster.cni_mtu, 1400);
    }

    #[test]
    fn cluster_validation_gates_on_enabled() {
        // Bad pod_cidr is rejected when enabled.
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=true\npod_cidr=\"not-a-cidr\"\n"
        )
        .is_err());
        // Unsupported distro is rejected.
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=true\ndistro=\"k3s\"\n"
        )
        .is_err());
        // control_plane_replicas must be 1/3/5 (etcd quorum); even/too-large is rejected.
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=true\ncontrol_plane_replicas=2\n"
        )
        .is_err());
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=true\ncontrol_plane_replicas=3\n"
        )
        .is_ok());
        // A zero provisioning timeout would degrade before any node could start; rejected.
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=true\nk8s_provisioning_timeout_secs=0\n"
        )
        .is_err());
        // Unknown storage provisioner is rejected; "none" and "local-path" are accepted.
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=true\nstorage_provisioner=\"nfs\"\n"
        )
        .is_err());
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=true\nstorage_provisioner=\"none\"\n"
        )
        .is_ok());
        // A local_path_dir that would break the JSON it is interpolated into is rejected.
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=true\nlocal_path_dir=\"/mnt/a\\\"b\"\n"
        )
        .is_err());
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=true\nlocal_path_dir=\"relative/path\"\n"
        )
        .is_err());
        // A local_path_dir only matters for local-path; junk is ignored when storage is "none".
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=true\nstorage_provisioner=\"none\"\nlocal_path_dir=\"bad path\"\n"
        )
        .is_ok());
        // Disabled cluster: no cluster validation applied even with junk values.
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=false\npod_cidr=\"whatever\"\n"
        )
        .is_ok());
    }

    #[test]
    fn cidr_overlap_detection() {
        assert!(cidrs_overlap("10.42.0.0/16", "10.42.5.0/24")); // /24 inside /16
        assert!(cidrs_overlap("10.42.0.0/16", "10.42.0.0/16")); // identical
        assert!(!cidrs_overlap("10.42.0.0/16", "10.43.0.0/16")); // adjacent, disjoint
        assert!(!cidrs_overlap("10.42.0.0/16", "10.96.0.0/12")); // pod vs default service
        assert!(!cidrs_overlap("bad", "10.42.0.0/16")); // malformed -> false
    }

    #[test]
    fn cluster_validation_rejects_overlapping_cidrs() {
        // pod_cidr overlapping service_cidr is rejected when enabled.
        assert!(SlurmConfig::load_from_str(
            "cluster_name=\"t\"\n[cluster]\nenabled=true\npod_cidr=\"10.42.0.0/16\"\nservice_cidr=\"10.42.5.0/24\"\n"
        )
        .is_err());
        // The defaults (10.42/16 pod vs 10.43/16 service) do NOT overlap.
        assert!(
            SlurmConfig::load_from_str("cluster_name=\"t\"\n[cluster]\nenabled=true\n").is_ok()
        );
    }

    #[test]
    fn test_controller_endpoints_multi_host() {
        let cfg = ControllerConfig {
            listen_addr: "[::]:6817".into(),
            hosts: vec!["ctrl1".into(), "ctrl2".into(), "ctrl3".into()],
            ..Default::default()
        };
        assert_eq!(
            cfg.endpoints(),
            vec![
                "http://ctrl1:6817",
                "http://ctrl2:6817",
                "http://ctrl3:6817"
            ]
        );
    }

    #[test]
    fn test_controller_endpoints_custom_port() {
        let cfg = ControllerConfig {
            listen_addr: "0.0.0.0:7000".into(),
            hosts: vec!["ctrl1".into()],
            ..Default::default()
        };
        assert_eq!(cfg.endpoints(), vec!["http://ctrl1:7000"]);
    }

    #[test]
    fn test_controller_endpoints_empty_hosts() {
        let cfg = ControllerConfig {
            hosts: vec![],
            ..Default::default()
        };
        assert!(cfg.endpoints().is_empty());
    }

    #[test]
    fn test_controller_endpoints_ipv6_host_bracketed() {
        let cfg = ControllerConfig {
            listen_addr: "[::]:6817".into(),
            hosts: vec!["::1".into()],
            ..Default::default()
        };
        assert_eq!(cfg.endpoints(), vec!["http://[::1]:6817"]);
    }

    #[test]
    fn test_controller_endpoints_ipv6_already_bracketed() {
        let cfg = ControllerConfig {
            listen_addr: "[::]:6817".into(),
            hosts: vec!["[::1]".into()],
            ..Default::default()
        };
        assert_eq!(cfg.endpoints(), vec!["http://[::1]:6817"]);
    }

    #[test]
    fn test_parse_time() {
        assert_eq!(parse_time_minutes("60"), Some(60));
        // MM:SS: 1m30s, rounded up to whole minutes.
        assert_eq!(parse_time_minutes("1:30"), Some(2));
        assert_eq!(parse_time_minutes("72:00:00"), Some(4320));
        assert_eq!(parse_time_minutes("1-00:00:00"), Some(1440));
        assert_eq!(parse_time_minutes("INFINITE"), None);
    }

    /// Every form in Slurm's documented `--time` grammar, in minutes:
    /// `minutes`, `minutes:seconds`, `hours:minutes:seconds`, `days-hours`,
    /// `days-hours:minutes`, `days-hours:minutes:seconds`.
    #[test]
    fn parse_time_minutes_covers_the_documented_slurm_grammar() {
        assert_eq!(parse_time_minutes("10"), Some(10));
        assert_eq!(parse_time_minutes("4:00"), Some(4));
        assert_eq!(parse_time_minutes("30:00"), Some(30));
        assert_eq!(parse_time_minutes("0:30"), Some(1));
        assert_eq!(parse_time_minutes("1:00:00"), Some(60));
        assert_eq!(parse_time_minutes("2-12"), Some(3600));
        assert_eq!(parse_time_minutes("1-0"), Some(1440));
        assert_eq!(parse_time_minutes("2-12:30"), Some(3630));
        assert_eq!(parse_time_minutes("2-12:30:15"), Some(3631));
    }

    /// The seconds field is not constrained to 0..59, so it must carry into
    /// whole minutes rather than contributing a flat one-minute round-up.
    #[test]
    fn parse_time_minutes_carries_an_unnormalised_seconds_field() {
        assert_eq!(parse_time_seconds("1:90"), Some(150));
        assert_eq!(parse_time_minutes("1:90"), Some(3)); // 150s
        assert_eq!(parse_time_minutes("0:59"), Some(1));
        assert_eq!(parse_time_minutes("0:60"), Some(1));
        assert_eq!(parse_time_minutes("0:61"), Some(2));
    }

    #[test]
    fn parse_time_minutes_carries_hms_seconds() {
        for (hms, seconds, minutes) in [
            ("0:0:0", 0, 0),
            ("0:0:59", 59, 1),
            ("0:0:60", 60, 1),
            ("0:0:61", 61, 2),
            ("0:0:90", 90, 2),
            ("0:0:120", 120, 2),
            ("1:2:121", 3_841, 65),
            ("0:0:4294967296", 4_294_967_296, 71_582_789),
        ] {
            for (input, expected_seconds, expected_minutes) in [
                (hms.to_string(), seconds, minutes),
                (format!("2-{hms}"), seconds + 172_800, minutes + 2_880),
            ] {
                assert_eq!(
                    parse_time_seconds(&input),
                    Some(expected_seconds),
                    "{input}"
                );
                assert_eq!(
                    parse_time_minutes(&input),
                    Some(expected_minutes),
                    "{input}"
                );
            }
        }
    }

    #[test]
    fn parse_time_minutes_checks_hms_overflow() {
        for prefix in ["", "0-"] {
            for hms in [
                "71582789:0:0",
                "0:4294967295:1",
                "0:0:257698037701",
                "5124095576030431:0:0",
                "0:307445734561825861:0",
                "1:0:18446744073709551615",
            ] {
                let input = format!("{prefix}{hms}");
                assert_eq!(parse_time_minutes(&input), None, "{input}");
            }
            let input = format!("{prefix}0:0:257698037640");
            assert_eq!(parse_time_minutes(&input), Some(u32::MAX - 1), "{input}");
        }
        assert_eq!(parse_time_minutes("2982616-4:14:61"), None);
    }

    #[test]
    fn test_parse_time_seconds() {
        // "N" → N minutes in seconds
        assert_eq!(parse_time_seconds("1"), Some(60));
        assert_eq!(parse_time_seconds("60"), Some(3600));
        // "MM:SS" → exact seconds
        assert_eq!(parse_time_seconds("1:30"), Some(90));
        // "H:MM:SS" → exact seconds (the key case)
        assert_eq!(parse_time_seconds("0:00:10"), Some(10));
        assert_eq!(parse_time_seconds("0:01:30"), Some(90));
        assert_eq!(parse_time_seconds("1:00:00"), Some(3600));
        // days-HH:MM:SS
        assert_eq!(parse_time_seconds("1-00:00:00"), Some(86400));
        assert_eq!(parse_time_seconds("7-00:00:00"), Some(604800));
        // limits
        assert_eq!(parse_time_seconds("INFINITE"), None);
        assert_eq!(parse_time_seconds("UNLIMITED"), None);
    }

    /// Every form in Slurm's documented `--time` grammar, in seconds. `4:00` is
    /// four minutes, not four hours, and `2-12` is a limit rather than no limit.
    #[test]
    fn parse_time_seconds_covers_the_documented_slurm_grammar() {
        assert_eq!(parse_time_seconds("10"), Some(600));
        assert_eq!(parse_time_seconds("4:00"), Some(240));
        assert_eq!(parse_time_seconds("30:00"), Some(1800));
        assert_eq!(parse_time_seconds("0:30"), Some(30));
        assert_eq!(parse_time_seconds("1:00:00"), Some(3600));
        assert_eq!(parse_time_seconds("0:00:10"), Some(10));
        assert_eq!(parse_time_seconds("2-12"), Some(216_000));
        assert_eq!(parse_time_seconds("1-0"), Some(86_400));
        assert_eq!(parse_time_seconds("2-12:30"), Some(217_800));
        assert_eq!(parse_time_seconds("2-12:30:15"), Some(217_815));
    }

    #[test]
    fn test_format_time() {
        assert_eq!(format_time(Some(90)), "01:30:00");
        assert_eq!(format_time(Some(1500)), "1-01:00:00");
        assert_eq!(format_time(None), "UNLIMITED");
    }

    #[test]
    fn test_load_metrics_config() {
        let toml = r#"
cluster_name = "test"

[metrics]
enabled = false
listen_addr = "[::]:9999"
bind = "all"
high_cardinality = true
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert!(!config.metrics.enabled);
        assert_eq!(config.metrics.listen_addr, "[::]:9999");
        assert_eq!(config.metrics.bind, MetricsBind::All);
        assert!(config.metrics.high_cardinality);
        assert_eq!(
            config.metrics.effective_listen_addr().unwrap(),
            "[::]:9999".parse().unwrap()
        );
    }

    /// A budget window and a priority-decay half-life are tuned for different
    /// reasons, so retuning one must never move the other.
    #[test]
    fn grp_wall_window_is_independent_of_the_fairshare_halflife() {
        let toml = r#"
cluster_name = "x"

[scheduler]
fairshare_halflife_days = 7
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert_eq!(config.scheduler.fairshare_halflife_days, 7);
        assert_eq!(config.accounting.grp_wall_window_days, 14);

        let toml = r#"
cluster_name = "x"

[accounting]
grp_wall_window_days = 30
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert_eq!(config.accounting.grp_wall_window_days, 30);
        assert_eq!(config.scheduler.fairshare_halflife_days, 14);
    }

    /// A zero window makes the consumption query match nothing, so every budget
    /// reads zero consumed and stops applying while `show qos` still displays the
    /// cap. An out-of-range value wraps negative through the `i32` bind and lands
    /// in the same place, so both are rejected at load.
    #[test]
    fn grp_wall_window_rejects_values_that_would_disable_enforcement() {
        let with_window = |days: &str| {
            format!(
                r#"
cluster_name = "x"

[accounting]
grp_wall_window_days = {days}
"#
            )
        };

        for bad in ["0", "3651", "4294967295"] {
            let err = SlurmConfig::load_from_str(&with_window(bad))
                .expect_err("a window that disables enforcement must not load");
            assert!(
                err.to_string().contains("accounting.grp_wall_window_days"),
                "error must name the offending field, got: {err}"
            );
        }

        for good in ["1", "3650"] {
            SlurmConfig::load_from_str(&with_window(good))
                .expect("a window inside the range must load");
        }
    }

    #[test]
    fn test_metrics_defaults() {
        let config = SlurmConfig::load_from_str(r#"cluster_name = "x""#).unwrap();
        assert!(config.metrics.enabled);
        assert_eq!(config.metrics.listen_addr, "[::]:6822");
        assert_eq!(config.metrics.bind, MetricsBind::Loopback);
        assert!(!config.metrics.high_cardinality);
        assert_eq!(
            config.metrics.effective_listen_addr().unwrap(),
            "127.0.0.1:6822".parse().unwrap()
        );
    }

    #[test]
    fn test_metrics_stale_exposition_format_ignored() {
        let toml = r#"
cluster_name = "x"

[metrics]
exposition_format = "slurm_0_0_4"
"#;
        let config = SlurmConfig::load_from_str(toml);
        assert!(
            config.is_ok(),
            "stale exposition_format key should be silently ignored"
        );
    }

    #[test]
    fn test_metrics_invalid_listen_addr() {
        let config = SlurmConfig::load_from_str(
            r#"
cluster_name = "x"

[metrics]
listen_addr = "not-a-socket"
"#,
        )
        .unwrap();
        assert!(config.metrics.effective_listen_addr().is_err());
    }

    #[test]
    fn test_load_hooks_config() {
        let toml = r#"
cluster_name = "test"

[hooks]
prolog = "/etc/spur/prolog.sh"
epilog = "/etc/spur/epilog.sh"
prolog_slurmctld = "/etc/spur/prolog_slurmctld.sh"
epilog_slurmctld = "/etc/spur/epilog_slurmctld.sh"
task_prolog = "/etc/spur/task_prolog.sh"
task_epilog = "/etc/spur/task_epilog.sh"
srun_prolog = "/etc/spur/srun_prolog.sh"
srun_epilog = "/etc/spur/srun_epilog.sh"
job_submit = "/etc/spur/job_submit.sh"
job_submit_lua = "/etc/spur/job_submit.lua"
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert_eq!(config.hooks.prolog.as_deref(), Some("/etc/spur/prolog.sh"));
        assert_eq!(config.hooks.epilog.as_deref(), Some("/etc/spur/epilog.sh"));
        assert_eq!(
            config.hooks.prolog_slurmctld.as_deref(),
            Some("/etc/spur/prolog_slurmctld.sh")
        );
        assert_eq!(
            config.hooks.epilog_slurmctld.as_deref(),
            Some("/etc/spur/epilog_slurmctld.sh")
        );
        assert_eq!(
            config.hooks.task_prolog.as_deref(),
            Some("/etc/spur/task_prolog.sh")
        );
        assert_eq!(
            config.hooks.task_epilog.as_deref(),
            Some("/etc/spur/task_epilog.sh")
        );
        assert_eq!(
            config.hooks.srun_prolog.as_deref(),
            Some("/etc/spur/srun_prolog.sh")
        );
        assert_eq!(
            config.hooks.srun_epilog.as_deref(),
            Some("/etc/spur/srun_epilog.sh")
        );
        assert_eq!(
            config.hooks.job_submit.as_deref(),
            Some("/etc/spur/job_submit.sh")
        );
        assert_eq!(
            config.hooks.job_submit_lua.as_deref(),
            Some("/etc/spur/job_submit.lua")
        );
        // metrics section omitted — should keep defaults
        assert!(config.metrics.enabled);
    }

    #[test]
    fn test_hooks_defaults() {
        let config = SlurmConfig::load_from_str(r#"cluster_name = "x""#).unwrap();
        assert!(config.hooks.prolog.is_none());
        assert!(config.hooks.epilog.is_none());
        assert!(config.hooks.prolog_slurmctld.is_none());
        assert!(config.hooks.epilog_slurmctld.is_none());
        assert!(config.hooks.task_prolog.is_none());
        assert!(config.hooks.task_epilog.is_none());
        assert!(config.hooks.srun_prolog.is_none());
        assert!(config.hooks.srun_epilog.is_none());
        assert!(config.hooks.job_submit.is_none());
        assert!(config.hooks.job_submit_lua.is_none());
        // hooks section omitted — metrics should keep defaults
        assert!(config.metrics.enabled);
        assert_eq!(config.metrics.listen_addr, "[::]:6822");
    }

    #[test]
    fn test_health_defaults() {
        // With no [health] section there are no checks, so the feature is off.
        let config = SlurmConfig::load_from_str(r#"cluster_name = "x""#).unwrap();
        assert!(config.health.checks.is_empty());
        assert!(!config.health.is_enabled());
    }

    #[test]
    fn test_health_parses_checks_and_per_check_defaults() {
        let toml = r#"
cluster_name = "x"

[[health.checks]]
program = "/etc/spur/gpu-health.sh"

[[health.checks]]
program = "/etc/spur/privileged-check.sh"
interval_secs = 120
timeout_secs = 15
max_wait_secs = 900
user = "root"
uid = 0
gid = 0
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert!(config.health.is_enabled());
        assert_eq!(config.health.checks.len(), 2);
        // Default cap protects against a fleet-wide wave without configuration.
        assert_eq!(config.health.max_unavailable, "10%");

        // First check takes the per-field defaults, including an unprivileged
        // identity (nobody:nogroup).
        let gpu = &config.health.checks[0];
        assert_eq!(gpu.program, "/etc/spur/gpu-health.sh");
        assert_eq!(gpu.nodes, 1);
        assert_eq!(gpu.interval_secs, 300);
        assert_eq!(gpu.timeout_secs, 60);
        assert_eq!(gpu.max_wait_secs, 600);
        assert_eq!(gpu.user, "nobody");
        assert_eq!(gpu.uid, 65534);
        assert_eq!(gpu.gid, 65534);

        // Second overrides them (a privileged check).
        let priv_check = &config.health.checks[1];
        assert_eq!(priv_check.program, "/etc/spur/privileged-check.sh");
        assert_eq!(priv_check.interval_secs, 120);
        assert_eq!(priv_check.timeout_secs, 15);
        assert_eq!(priv_check.max_wait_secs, 900);
        assert_eq!(priv_check.user, "root");
        assert_eq!(priv_check.uid, 0);
        assert_eq!(priv_check.gid, 0);
    }

    #[test]
    fn test_health_rejects_relative_program() {
        // A relative path would resolve to nothing on the node and drain it, so
        // it is caught at config load instead.
        let err = SlurmConfig::load_from_str(
            "cluster_name = \"x\"\n[[health.checks]]\nprogram = \"gpu-health.sh\"\n",
        )
        .unwrap_err();
        assert!(
            matches!(err, ConfigError::InvalidValue { field, .. } if field.contains("program"))
        );
    }

    #[test]
    fn test_health_rejects_multi_node_until_topology() {
        let err = SlurmConfig::load_from_str(
            "cluster_name = \"x\"\n[[health.checks]]\nprogram = \"/x.sh\"\nnodes = 2\n",
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::InvalidValue { field, .. } if field.contains("nodes")));
    }

    #[test]
    fn test_health_max_unavailable_parses_and_validates() {
        // Absolute and percentage forms both parse; a bad value is rejected.
        assert_eq!(parse_health_cap("5", 40), Some(5));
        assert_eq!(parse_health_cap("10%", 40), Some(4));
        assert_eq!(parse_health_cap("10%", 3), Some(1)); // rounds up, at least 1
        assert_eq!(parse_health_cap("0", 40), Some(0)); // unlimited
        assert_eq!(parse_health_cap("0%", 40), Some(0));
        assert_eq!(parse_health_cap("bogus", 40), None);

        let err = SlurmConfig::load_from_str(
            "cluster_name = \"x\"\n[health]\nmax_unavailable = \"lots\"\n",
        )
        .unwrap_err();
        assert!(
            matches!(err, ConfigError::InvalidValue { field, .. } if field == "health.max_unavailable")
        );
    }

    #[test]
    fn test_load_config() {
        let toml = r#"
cluster_name = "test-cluster"

[controller]
listen_addr = "[::]:6817"
rest_addr = "[::]:6820"
hosts = ["ctrl1", "ctrl2"]
state_dir = "/var/spool/spur"
max_job_id = 999999999
first_job_id = 100

[scheduler]
plugin = "backfill"
interval_secs = 2

[[partitions]]
name = "gpu"
default = true
nodes = "gpu[001-008]"
max_time = "72:00:00"

[[partitions]]
name = "cpu"
nodes = "cpu[001-064]"
max_time = "168:00:00"

[[nodes]]
names = "gpu[001-008]"
cpus = 128
memory_mb = 512000
gres = ["gpu:mi300x:8"]

[[nodes]]
names = "cpu[001-064]"
cpus = 256
memory_mb = 1024000
"#;

        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert_eq!(config.cluster_name, "test-cluster");
        assert_eq!(config.partitions.len(), 2);
        assert_eq!(config.nodes.len(), 2);
        assert!(config.partitions[0].default);

        let parts = config.build_partitions();
        assert_eq!(parts[0].name, "gpu");
        assert_eq!(parts[0].max_time_minutes, Some(4320));
    }

    #[test]
    fn parse_partition_time_distinguishes_unlimited_from_invalid() {
        assert_eq!(
            parse_partition_time("p.max_time", "UNLIMITED").unwrap(),
            None
        );
        assert_eq!(
            parse_partition_time("p.max_time", "infinite").unwrap(),
            None
        );
        assert_eq!(
            parse_partition_time("p.max_time", "01:00:00").unwrap(),
            Some(60)
        );
        // Suffixed durations are accepted (honor intent instead of failing).
        assert_eq!(parse_partition_time("p.max_time", "1h").unwrap(), Some(60));
        assert_eq!(parse_partition_time("p.max_time", "90m").unwrap(), Some(90));
        // A genuine typo still fails loudly.
        assert!(parse_partition_time("p.max_time", "1 hour").is_err());
        assert!(parse_partition_time("p.max_time", "banana").is_err());
    }

    #[test]
    fn enforce_part_limits_parses_case_insensitively_and_yes_alias() {
        #[derive(serde::Deserialize)]
        struct W {
            v: EnforcePartLimits,
        }
        let parse = |v: &str| toml::from_str::<W>(&format!("v = \"{v}\"")).unwrap().v;
        assert_eq!(parse("no"), EnforcePartLimits::No);
        assert_eq!(parse("all"), EnforcePartLimits::All);
        assert_eq!(parse("ALL"), EnforcePartLimits::All);
        assert_eq!(parse("Yes"), EnforcePartLimits::All);
        assert_eq!(parse("any"), EnforcePartLimits::Any);
        assert_eq!(parse("ANY"), EnforcePartLimits::Any);
    }

    #[test]
    fn enforce_part_limits_rejects_unknown_value() {
        #[derive(serde::Deserialize)]
        struct W {
            #[allow(dead_code)]
            v: EnforcePartLimits,
        }
        assert!(toml::from_str::<W>("v = \"sometimes\"").is_err());
    }

    #[test]
    fn format_time_seconds_keeps_second_precision() {
        assert_eq!(format_time_seconds(Some(630)), "00:10:30");
        assert_eq!(format_time_seconds(Some(3600)), "01:00:00");
        assert_eq!(format_time_seconds(Some(90_061)), "1-01:01:01");
        assert_eq!(format_time_seconds(None), "UNLIMITED");
    }

    #[test]
    fn parse_time_minutes_accepts_suffixed_durations() {
        assert_eq!(parse_time_minutes("1h"), Some(60));
        assert_eq!(parse_time_minutes("90m"), Some(90));
        assert_eq!(parse_time_minutes("1h40m"), Some(100));
        assert_eq!(parse_time_minutes("2d"), Some(2880));
        assert_eq!(parse_time_minutes("1d12h30m"), Some(2190));
        // Sub-minute remainders round up, matching the "H:MM:SS" seconds rule.
        assert_eq!(parse_time_minutes("30s"), Some(1));
        assert_eq!(parse_time_minutes("1h15s"), Some(61));
        // Slurm grammar is unchanged and still takes precedence.
        assert_eq!(parse_time_minutes("60"), Some(60));
        assert_eq!(parse_time_minutes("01:00:00"), Some(60));
        assert_eq!(parse_time_minutes("1-00:00:00"), Some(1440));
        // Non-durations are rejected, not silently accepted.
        assert_eq!(parse_time_minutes("1 hour"), None);
        assert_eq!(parse_time_minutes("1.5h"), None);
        assert_eq!(parse_time_minutes("1h30"), None);
        assert_eq!(parse_time_minutes("h"), None);
        assert_eq!(parse_time_minutes("1x"), None);
    }

    #[test]
    fn parse_time_seconds_accepts_suffixed_durations() {
        assert_eq!(parse_time_seconds("1h"), Some(3600));
        assert_eq!(parse_time_seconds("90m"), Some(5400));
        assert_eq!(parse_time_seconds("30s"), Some(30));
        assert_eq!(parse_time_seconds("1h30m15s"), Some(5415));
        // Slurm grammar unchanged.
        assert_eq!(parse_time_seconds("60"), Some(3600));
        assert_eq!(parse_time_seconds("00:00:30"), Some(30));
    }

    #[test]
    fn parse_time_rejects_overflowing_slurm_durations() {
        // u32 minutes overflow: 2982617 days * 1440 is u32::MAX + 1185.
        assert_eq!(parse_time_minutes("2982617-00:00:00"), None);
        // Overflow on the `+ hms`, not the multiply: 2982616 days leaves only
        // 255 minutes of headroom and "05:00:00" is 300.
        assert_eq!(parse_time_minutes("2982616-05:00:00"), None);
        // MM:SS arm. Minutes are now taken directly rather than multiplied, so
        // overflow needs the sub-minute round-up to push past u32::MAX.
        assert_eq!(parse_time_minutes("4294967295:01"), None);
        // A minute count that does not fit u32 at all is rejected by the parse.
        assert_eq!(parse_time_minutes("4294967296:00"), None);
        // u64 seconds overflow.
        assert_eq!(parse_time_seconds("213503982334602-00:00:00"), None);
        // The largest non-overflowing value is still accepted unchanged.
        assert_eq!(parse_time_minutes("2982616-00:00:00"), Some(4_294_967_040));
    }

    #[test]
    fn parse_partition_time_rejects_overflowing_duration() {
        // validate() promises a bad partition time fails loudly rather than
        // silently enforcing a different cap.
        assert!(matches!(
            parse_partition_time("partitions.gpu.max_time", "2982617-00:00:00"),
            Err(ConfigError::InvalidValue { .. })
        ));
    }

    #[test]
    fn load_accepts_suffixed_partition_max_time() {
        // "1h" is honored as a 60-minute cap rather than failing or silently
        // becoming UNLIMITED.
        let toml = r#"
cluster_name = "test"

[[partitions]]
name = "gpu"
max_time = "1h"
default_time = "30m"
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        let parts = config.build_partitions();
        assert_eq!(parts[0].max_time_minutes, Some(60));
        assert_eq!(parts[0].default_time_minutes, Some(30));
    }

    #[test]
    fn load_rejects_invalid_partition_max_time() {
        // A value that is neither Slurm grammar nor a suffixed duration must
        // fail loudly rather than silently becoming UNLIMITED.
        let toml = r#"
cluster_name = "test"

[[partitions]]
name = "gpu"
max_time = "1 hour"
"#;
        let err = SlurmConfig::load_from_str(toml).unwrap_err();
        assert!(
            matches!(err, ConfigError::InvalidValue { ref field, .. } if field.contains("max_time")),
            "expected InvalidValue for max_time, got: {err:?}"
        );
    }

    #[test]
    fn load_accepts_unlimited_partition_max_time() {
        let toml = r#"
cluster_name = "test"

[[partitions]]
name = "gpu"
max_time = "UNLIMITED"
default_time = "INFINITE"
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        let parts = config.build_partitions();
        assert_eq!(parts[0].max_time_minutes, None);
        assert_eq!(parts[0].default_time_minutes, None);
    }

    #[test]
    fn load_rejects_default_time_above_max_time() {
        // A finite DefaultTime over MaxTime would auto-fill a -t-less job past the
        // partition's own cap, pending it forever. Reject at load.
        let toml = r#"
cluster_name = "test"

[[partitions]]
name = "gpu"
max_time = "01:00:00"
default_time = "02:00:00"
"#;
        let err = SlurmConfig::load_from_str(toml).unwrap_err();
        assert!(
            matches!(err, ConfigError::InvalidValue { ref field, .. } if field.contains("default_time")),
            "expected InvalidValue for default_time, got: {err:?}"
        );
    }

    #[test]
    fn load_accepts_default_time_equal_to_max_time() {
        let toml = r#"
cluster_name = "test"

[[partitions]]
name = "gpu"
max_time = "01:00:00"
default_time = "01:00:00"
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        let parts = config.build_partitions();
        assert_eq!(parts[0].default_time_minutes, Some(60));
    }

    #[test]
    fn load_accepts_default_time_with_unlimited_max_time() {
        // UNLIMITED MaxTime caps nothing, so any finite DefaultTime is coherent.
        let toml = r#"
cluster_name = "test"

[[partitions]]
name = "gpu"
max_time = "UNLIMITED"
default_time = "02:00:00"
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        let parts = config.build_partitions();
        assert_eq!(parts[0].max_time_minutes, None);
        assert_eq!(parts[0].default_time_minutes, Some(120));
    }

    #[test]
    fn build_partitions_propagates_partition_access_control() {
        let toml = r#"
cluster_name = "test"

[[partitions]]
name = "gpu"
allow_accounts = ["research", "faculty"]
deny_accounts = ["student"]
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        let parts = config.build_partitions();
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0].allow_accounts,
            vec!["research".to_string(), "faculty".to_string()]
        );
        assert_eq!(parts[0].deny_accounts, vec!["student".to_string()]);
    }

    #[test]
    fn controller_config_defaults_for_new_fields() {
        let config = ControllerConfig::default();
        assert_eq!(config.heartbeat_timeout_secs, None);
        assert_eq!(config.max_batch_requeue, 5);
        assert_eq!(config.max_launch_backoff_secs, 300);
        assert!(config.hold_on_prolog_fail);
    }

    #[test]
    fn controller_config_parses_hold_on_prolog_fail() {
        let toml = r#"
cluster_name = "test"

[controller]
hold_on_prolog_fail = false
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert!(!config.controller.hold_on_prolog_fail);
    }

    #[test]
    fn controller_config_parses_max_batch_requeue() {
        let toml = r#"
cluster_name = "test"

[controller]
max_batch_requeue = 7
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert_eq!(config.controller.max_batch_requeue, 7);
    }

    #[test]
    fn scheduler_config_rejects_inactive_limit_below_floor() {
        let toml = r#"
cluster_name = "test"

[scheduler]
inactive_limit_secs = 20
"#;
        let err = SlurmConfig::load_from_str(toml).unwrap_err();
        assert!(
            err.to_string().contains("inactive_limit_secs"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn scheduler_config_accepts_disabled_or_ample_inactive_limit() {
        // 0 disables reaping and is always allowed.
        SlurmConfig::load_from_str(
            "cluster_name = \"test\"\n[scheduler]\ninactive_limit_secs = 0\n",
        )
        .expect("0 (disabled) must be accepted");
        // A value at the floor (2x the keepalive interval) is accepted.
        SlurmConfig::load_from_str(
            "cluster_name = \"test\"\n[scheduler]\ninactive_limit_secs = 60\n",
        )
        .expect("a value at the floor must be accepted");
    }

    #[test]
    fn controller_config_rejects_zero_max_batch_requeue() {
        let toml = r#"
cluster_name = "test"

[controller]
max_batch_requeue = 0
"#;
        let err = SlurmConfig::load_from_str(toml).unwrap_err();
        assert!(
            err.to_string().contains("max_batch_requeue"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn auth_required_without_jwt_key_is_refused() {
        // `required` with no signing key would fall back to a public constant that anyone can forge
        // an admin identity against — strictly worse than the default. Startup must refuse it.
        let toml = r#"
cluster_name = "test"

[auth]
plugin = "jwt"
mode = "required"
"#;
        let err = SlurmConfig::load_from_str(toml).unwrap_err();
        assert!(
            err.to_string().contains("jwt_key"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn auth_required_with_empty_jwt_key_is_refused() {
        let toml = r#"
cluster_name = "test"

[auth]
plugin = "jwt"
mode = "required"
jwt_key = ""
"#;
        let err = SlurmConfig::load_from_str(toml).unwrap_err();
        assert!(
            err.to_string().contains("jwt_key"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn auth_required_with_jwt_key_is_accepted() {
        let toml = r#"
cluster_name = "test"

[auth]
plugin = "jwt"
mode = "required"
jwt_key = "a-real-secret"
"#;
        let config = SlurmConfig::load_from_str(toml).expect("required + a key must be accepted");
        assert_eq!(config.auth.mode, AuthMode::Required);
        assert_eq!(config.auth.jwt_key.as_deref(), Some("a-real-secret"));
    }

    #[test]
    fn auth_permissive_without_jwt_key_is_accepted() {
        // Only `required` is refused key-less; permissive must still come up so a cluster can adopt
        // authentication incrementally.
        let toml = r#"
cluster_name = "test"

[auth]
plugin = "jwt"
mode = "permissive"
"#;
        let config = SlurmConfig::load_from_str(toml).expect("permissive must not require a key");
        assert_eq!(config.auth.mode, AuthMode::Permissive);
    }

    #[test]
    fn controller_config_parses_max_launch_backoff_secs() {
        let toml = r#"
cluster_name = "test"

[controller]
max_launch_backoff_secs = 90
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert_eq!(config.controller.max_launch_backoff_secs, 90);
    }

    #[test]
    fn controller_config_rejects_out_of_range_max_launch_backoff_secs() {
        // Past this bound the computed hold instant overflows and panics the
        // controller, so it must be refused at load rather than at first use.
        let toml = format!(
            r#"
cluster_name = "test"

[controller]
max_launch_backoff_secs = {}
"#,
            MAX_LAUNCH_BACKOFF_SECS + 1
        );
        let err = SlurmConfig::load_from_str(&toml).unwrap_err();
        assert!(
            err.to_string().contains("max_launch_backoff_secs"),
            "unexpected error: {err}"
        );

        let ok = format!(
            r#"
cluster_name = "test"

[controller]
max_launch_backoff_secs = {MAX_LAUNCH_BACKOFF_SECS}
"#
        );
        assert_eq!(
            SlurmConfig::load_from_str(&ok)
                .unwrap()
                .controller
                .max_launch_backoff_secs,
            MAX_LAUNCH_BACKOFF_SECS,
            "the bound itself must be accepted"
        );
    }

    #[test]
    fn controller_config_rejects_zero_max_launch_backoff_secs() {
        // Zero would clamp every hold to zero and restore the retry storm.
        let toml = r#"
cluster_name = "test"

[controller]
max_launch_backoff_secs = 0
"#;
        let err = SlurmConfig::load_from_str(toml).unwrap_err();
        assert!(
            err.to_string().contains("max_launch_backoff_secs"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn controller_config_rejects_out_of_range_terminal_job_retention_secs() {
        // Beyond the guardrail, load must reject up front rather than silently
        // retaining terminal jobs long enough to defeat the memory bound.
        let toml = format!(
            r#"
cluster_name = "test"

[controller]
terminal_job_retention_secs = {}
"#,
            MAX_TERMINAL_JOB_RETENTION_SECS + 1
        );
        let err = SlurmConfig::load_from_str(&toml).unwrap_err();
        assert!(
            err.to_string().contains("terminal_job_retention_secs"),
            "unexpected error: {err}"
        );

        let ok = format!(
            r#"
cluster_name = "test"

[controller]
terminal_job_retention_secs = {MAX_TERMINAL_JOB_RETENTION_SECS}
"#
        );
        assert_eq!(
            SlurmConfig::load_from_str(&ok)
                .unwrap()
                .controller
                .terminal_job_retention_secs,
            MAX_TERMINAL_JOB_RETENTION_SECS,
            "the bound itself must be accepted"
        );
    }

    #[test]
    fn controller_config_rejects_out_of_range_agent_channel_timeouts() {
        for field in [
            "agent_connect_timeout_secs",
            "agent_keepalive_interval_secs",
            "agent_keepalive_timeout_secs",
        ] {
            let toml = format!(
                "cluster_name = \"test\"\n\n[controller]\n{field} = {}\n",
                MAX_AGENT_CHANNEL_TIMEOUT_SECS + 1
            );
            let err = SlurmConfig::load_from_str(&toml).unwrap_err();
            assert!(
                err.to_string().contains(field),
                "{field} past the ceiling must be rejected, got: {err}"
            );
        }
    }

    #[test]
    fn controller_config_rejects_a_zero_keepalive_timeout_while_keepalive_is_on() {
        let toml = r#"
cluster_name = "test"

[controller]
agent_keepalive_interval_secs = 10
agent_keepalive_timeout_secs = 0
"#;
        let err = SlurmConfig::load_from_str(toml).unwrap_err();
        assert!(
            err.to_string().contains("agent_keepalive_timeout_secs"),
            "a zero ping timeout marks every ping overdue; it must be rejected: {err}"
        );

        // Harmless once keepalive itself is off.
        let ok = r#"
cluster_name = "test"

[controller]
agent_keepalive_interval_secs = 0
agent_keepalive_timeout_secs = 0
"#;
        assert!(SlurmConfig::load_from_str(ok).is_ok());
    }

    #[test]
    fn controller_config_rejects_out_of_range_dispatch_reject_cooldown_secs() {
        // Past this bound load must reject rather than accept an unbounded
        // dispatch cooldown. Zero is valid (disables the cooldown).
        let toml = format!(
            r#"
cluster_name = "test"

[controller]
dispatch_reject_cooldown_secs = {}
"#,
            MAX_LAUNCH_BACKOFF_SECS + 1
        );
        let err = SlurmConfig::load_from_str(&toml).unwrap_err();
        assert!(
            err.to_string().contains("dispatch_reject_cooldown_secs"),
            "unexpected error: {err}"
        );

        let ok = r#"
cluster_name = "test"

[controller]
dispatch_reject_cooldown_secs = 0
"#;
        assert_eq!(
            SlurmConfig::load_from_str(ok)
                .unwrap()
                .controller
                .dispatch_reject_cooldown_secs,
            0,
            "zero (disabled) must be accepted"
        );
    }

    #[test]
    fn controller_config_parses_heartbeat_timeout() {
        let toml = r#"
cluster_name = "test"

[controller]
heartbeat_timeout_secs = 120
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert_eq!(config.controller.heartbeat_timeout_secs, Some(120));
    }

    #[test]
    fn controller_config_absent_fields_are_none() {
        let toml = r#"
cluster_name = "test"

[controller]
listen_addr = "[::]:6817"
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert_eq!(config.controller.heartbeat_timeout_secs, None);
    }

    #[test]
    fn rlimits_default_is_unlimited() {
        let cfg = RlimitsConfig::default();
        assert_eq!(cfg.memlock_limit().unwrap(), MemlockLimit::Unlimited);
    }

    #[test]
    fn rlimits_parses_unlimited() {
        let cfg = RlimitsConfig {
            memlock: "unlimited".into(),
        };
        assert_eq!(cfg.memlock_limit().unwrap(), MemlockLimit::Unlimited);
    }

    #[test]
    fn rlimits_parses_inherit() {
        let cfg = RlimitsConfig {
            memlock: "inherit".into(),
        };
        assert_eq!(cfg.memlock_limit().unwrap(), MemlockLimit::Inherit);
    }

    #[test]
    fn rlimits_parses_bytes() {
        let cfg = RlimitsConfig {
            memlock: "1048576".into(),
        };
        assert_eq!(cfg.memlock_limit().unwrap(), MemlockLimit::Bytes(1048576));
    }

    #[test]
    fn rlimits_invalid_errors() {
        let cfg = RlimitsConfig {
            memlock: "bogus".into(),
        };
        assert!(cfg.memlock_limit().is_err());
    }

    #[test]
    fn rlimits_zero_treated_as_unlimited() {
        let cfg = RlimitsConfig {
            memlock: "0".into(),
        };
        assert_eq!(cfg.memlock_limit().unwrap(), MemlockLimit::Unlimited);
    }

    #[test]
    fn rlimits_from_toml_default() {
        let toml = r#"
cluster_name = "test"
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert_eq!(
            config.rlimits.memlock_limit().unwrap(),
            MemlockLimit::Unlimited
        );
    }

    #[test]
    fn rlimits_from_toml_explicit() {
        let toml = r#"
cluster_name = "test"

[rlimits]
memlock = "inherit"
"#;
        let config = SlurmConfig::load_from_str(toml).unwrap();
        assert_eq!(
            config.rlimits.memlock_limit().unwrap(),
            MemlockLimit::Inherit
        );
    }
}
