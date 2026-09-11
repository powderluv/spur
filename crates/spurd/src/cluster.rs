// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Spurd-owned k0s systemd unit supervisor.
//!
//! SPUR owns the k0s lifecycle by OWNING a systemd unit for `k0s controller` / `k0s worker`
//! rather than forking k0s as a spurd child job. As a consequence k0s:
//!   * survives a spurd restart (it is not a spurd-parented process, so its delegated
//!     containerd/kubelet/pod cgroup is not orphaned when spurd exits), and
//!   * stays entirely out of the executor / `start_monitor` (2s reaper) / `enforce_time_limits`
//!     (SIGTERM->SIGKILL) job path, which would otherwise drop or kill a long-running cluster.
//!
//! Recovery is split between systemd and this supervisor (validated on a real host):
//!   * process crash / `systemctl kill`     -> systemd's own `Restart=always` re-spawns it.
//!   * unit deleted / disabled / stopped / config-drifted -> this supervisor's reconcile loop
//!     re-writes + re-enables + re-starts the unit.
//!
//! The unit uses the same directives k0s's own installer emits: `Type=simple` (k0s does NOT
//! `sd_notify`, so `Type=notify` would hang forever in `activating`), `KillMode=process`
//! (deliberately leaves the delegated containerd/kubelet/pod cgroup alone on stop, unlike
//! `KillMode=mixed` which would SIGKILL it), `Delegate=yes`, and `Restart=always`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Which k0s role this node's spurd-owned unit runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterRole {
    /// `k0s controller` — control-plane only (head node).
    Controller,
    /// `k0s worker` — GPU worker; requires a join token.
    Worker,
    /// `k0s controller --single` — all-in-one control-plane+worker (dev / single-node).
    Single,
}

impl ClusterRole {
    /// systemd unit base name (matches k0s's own convention).
    fn unit_name(self) -> &'static str {
        match self {
            ClusterRole::Worker => "k0sworker",
            ClusterRole::Controller | ClusterRole::Single => "k0scontroller",
        }
    }

    /// `k0s` subcommand + role flags for `ExecStart`.
    fn exec_args(self) -> &'static str {
        match self {
            ClusterRole::Controller => "controller",
            ClusterRole::Worker => "worker",
            ClusterRole::Single => "controller --single=true",
        }
    }

    /// Wire string for the role (matches the controller's `StartClusterComponentRequest.role`).
    pub fn as_str(self) -> &'static str {
        match self {
            ClusterRole::Controller => "controller",
            ClusterRole::Worker => "worker",
            ClusterRole::Single => "single",
        }
    }

    /// Parse the controller-supplied role string.
    pub fn parse_role(s: &str) -> Option<Self> {
        match s {
            "controller" => Some(ClusterRole::Controller),
            "worker" => Some(ClusterRole::Worker),
            "single" => Some(ClusterRole::Single),
            _ => None,
        }
    }
}

/// Inputs the supervisor needs to own a k0s unit.
///
/// Increment-2 prototype: sourced from env (`from_env`). Later increments populate this from
/// `spur_core` `ClusterConfig` + a controller-supplied role/join-token over the agent RPC.
#[derive(Clone, Debug)]
pub struct ClusterConfig {
    pub role: ClusterRole,
    /// Path to the `k0s` binary (`ConditionFileIsExecutable` + `ExecStart`).
    pub k0s_binary: PathBuf,
    /// `--config` path the unit passes to k0s (None for `--single` dev).
    pub k0s_config: Option<PathBuf>,
    /// `--token-file` for a worker join (None for a controller).
    pub join_token_file: Option<PathBuf>,
    /// Worker kubelet `--node-ip` (its mesh IP) for a native-routing CNI. None = k0s default.
    pub node_ip: Option<String>,
}

/// Owns and reconciles a single spurd-managed k0s systemd unit.
///
/// `Clone` so `K0sAgent` can snapshot the current supervisor out from under its `active` mutex and
/// then do the (blocking, seconds-long) systemctl/k0s IO WITHOUT holding the lock — otherwise a
/// stop/reconcile would serialize concurrent status/start RPCs behind it.
#[derive(Clone)]
pub struct ClusterSupervisor {
    cfg: ClusterConfig,
    /// systemd unit directory (overridable in tests; `/etc/systemd/system` in production).
    unit_dir: PathBuf,
}

impl ClusterSupervisor {
    pub fn new(cfg: ClusterConfig) -> Self {
        Self {
            cfg,
            unit_dir: PathBuf::from("/etc/systemd/system"),
        }
    }

    fn unit_file_name(&self) -> String {
        format!("{}.service", self.cfg.role.unit_name())
    }

    fn unit_path(&self) -> PathBuf {
        self.unit_dir.join(self.unit_file_name())
    }

    /// Render the unit file. Mirrors k0s's own installer directives (see module docs for why
    /// `Type=simple` / `KillMode=process` and not `notify` / `mixed`). A finite
    /// `StartLimitIntervalSec`/`StartLimitBurst` bounds a hard crash-loop (a genuinely broken node
    /// stops hammering every `RestartSec`); the reconcile supervisor is the higher-level recovery
    /// authority and re-enables the unit on its next tick, so the limit adds backoff without a
    /// permanent lockout.
    fn render_unit(&self) -> String {
        let mut exec = format!(
            "{} {}",
            self.cfg.k0s_binary.display(),
            self.cfg.role.exec_args()
        );
        if let Some(cfg) = &self.cfg.k0s_config {
            exec.push_str(&format!(" --config={}", cfg.display()));
        }
        if let Some(tok) = &self.cfg.join_token_file {
            exec.push_str(&format!(" --token-file={}", tok.display()));
        }
        if let Some(ip) = &self.cfg.node_ip {
            // Pin kubelet's node IP to the mesh IP so a native-routing CNI peers over the mesh.
            exec.push_str(&format!(" --kubelet-extra-args=--node-ip={ip}"));
        }
        format!(
            "[Unit]\n\
             Description=k0s {role} (managed by spurd)\n\
             Documentation=https://docs.k0sproject.io\n\
             ConditionFileIsExecutable={bin}\n\
             After=network-online.target\n\
             Wants=network-online.target\n\
             \n\
             [Service]\n\
             ExecStart={exec}\n\
             Restart=always\n\
             RestartSec=10\n\
             StartLimitIntervalSec=300\n\
             StartLimitBurst=5\n\
             Delegate=yes\n\
             KillMode=process\n\
             LimitCORE=infinity\n\
             LimitNOFILE=999999\n\
             TasksMax=infinity\n\
             TimeoutStartSec=0\n\
             \n\
             [Install]\n\
             WantedBy=multi-user.target\n",
            role = self.cfg.role.unit_name(),
            bin = self.cfg.k0s_binary.display(),
            exec = exec,
        )
    }

    /// One reconcile pass. Returns `Ok(true)` if it changed anything (wrote/enabled/started).
    /// Idempotent: a steady-state healthy unit is a no-op.
    pub async fn reconcile_once(&self) -> anyhow::Result<bool> {
        // Distinguish "cannot reach systemd" (transient/environment) from "k0s degraded".
        if !self.systemd_available().await {
            warn!("systemctl unavailable; skipping cluster reconcile this tick");
            return Ok(false);
        }

        let mut changed = false;

        // 1. Ensure the unit file exists and matches desired content.
        let desired = self.render_unit();
        let path = self.unit_path();
        let needs_write = match tokio::fs::read_to_string(&path).await {
            Ok(current) => current != desired,
            Err(_) => true,
        };
        if needs_write {
            info!(unit = %path.display(), "writing/updating spurd-owned k0s unit file");
            self.write_unit_atomic(&path, &desired).await?;
            self.systemctl(&["daemon-reload"]).await?;
            changed = true;
        }

        // 2. Ensure enabled (survives reboot).
        if !self.unit_status_is(&["is-enabled"], "enabled").await {
            info!(unit = %self.unit_file_name(), "enabling unit");
            self.systemctl(&["enable", &self.unit_file_name()]).await?;
            changed = true;
        }

        // 3. Ensure active. systemd `Restart=always` handles process crashes; this catches a unit
        //    that was stopped/disabled out-of-band (which Restart= does not cover).
        if !self.unit_status_is(&["is-active"], "active").await {
            info!(unit = %self.unit_file_name(), "unit not active; starting");
            self.systemctl(&["start", &self.unit_file_name()]).await?;
            changed = true;
        }

        Ok(changed)
    }

    async fn systemd_available(&self) -> bool {
        Command::new("systemctl")
            .arg("--version")
            .output()
            .await
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Atomic write: temp file in the same dir, then rename (containerd/systemd never see a
    /// partial unit file).
    async fn write_unit_atomic(&self, path: &Path, content: &str) -> anyhow::Result<()> {
        let mut tmp = OsString::from(path.as_os_str());
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        tokio::fs::write(&tmp, content).await?;
        tokio::fs::rename(&tmp, path).await?;
        Ok(())
    }

    async fn systemctl(&self, args: &[&str]) -> anyhow::Result<()> {
        let out = Command::new("systemctl").args(args).output().await?;
        if !out.status.success() {
            anyhow::bail!(
                "systemctl {:?} failed: {}",
                args,
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(())
    }

    /// `systemctl <verb> <unit>` and compare the trimmed stdout to `expect`
    /// (e.g. is-active -> "active", is-enabled -> "enabled").
    async fn unit_status_is(&self, verb: &[&str], expect: &str) -> bool {
        let unit = self.unit_file_name();
        let mut args: Vec<&str> = verb.to_vec();
        args.push(&unit);
        Command::new("systemctl")
            .args(&args)
            .output()
            .await
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == expect)
            .unwrap_or(false)
    }

    /// systemd active-state of the unit ("active"/"inactive"/"failed"/"activating"/"unknown").
    async fn component_state(&self) -> String {
        let unit = self.unit_file_name();
        Command::new("systemctl")
            .args(["show", &unit, "-p", "ActiveState", "--value"])
            .output()
            .await
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// systemd `NRestarts` for the unit (cumulative auto-restarts); 0 when unknown/unavailable.
    async fn unit_restart_count(&self) -> u64 {
        let unit = self.unit_file_name();
        Command::new("systemctl")
            .args(["show", &unit, "-p", "NRestarts", "--value"])
            .output()
            .await
            .ok()
            .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
            .unwrap_or(0)
    }

    fn role(&self) -> ClusterRole {
        self.cfg.role
    }

    /// Parse this unit's rendered `ExecStart` back into a `ClusterConfig` (recovering the exact
    /// role — incl. `Single` via `--single` — the k0s binary, and any `--config`/`--token-file`
    /// paths), so a rebuilt supervisor re-renders an identical unit and reconcile stays a no-op.
    async fn read_cfg_from_unit(&self) -> anyhow::Result<ClusterConfig> {
        let path = self.unit_path();
        let content = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("read unit file {}", path.display()))?;
        let exec = content
            .lines()
            .find_map(|l| l.trim().strip_prefix("ExecStart="))
            .context("unit file has no ExecStart= line")?;
        parse_execstart(exec)
    }

    /// Stop + disable the unit; optionally `k0s reset` (destructive) followed by purging the
    /// spurd-owned unit file + join token so nothing restarts the component. Idempotent.
    async fn stop(&self, reset: bool) -> anyhow::Result<()> {
        if !self.systemd_available().await {
            anyhow::bail!("systemctl unavailable");
        }
        self.disable_unit().await;
        if reset {
            self.reset_and_purge().await?;
        }
        Ok(())
    }

    /// `k0s reset` (wipes /var/lib/k0s) followed by `purge_unit` (removes the spurd-owned unit +
    /// join token). Purges UNCONDITIONALLY, even if `k0s reset` fails (e.g. a broken/half-installed
    /// k0s binary): a surviving token was minted against the torn-down CA and makes the node fail its
    /// next join with a kubernetes-ca verification error, while the `Restart=always` unit keeps
    /// restarting k0s with it. The reset failure is surfaced, but only after the purge.
    async fn reset_and_purge(&self) -> anyhow::Result<()> {
        let reset_result = self.k0s_reset().await;
        self.purge_unit().await;
        reset_result
    }

    /// `systemctl disable --now <unit>` — stop + remove the enable symlink. Idempotent (ignores an
    /// absent unit).
    async fn disable_unit(&self) {
        let unit = self.unit_file_name();
        let _ = self.systemctl(&["disable", "--now", &unit]).await;
    }

    /// Delete the spurd-owned unit file + join token and reload systemd, so nothing (Restart=always,
    /// or a reconcile that re-adopts a lingering unit) can bring the component back. Best-effort: an
    /// already-absent file is fine. Call only when tearing a node down for good (reset path).
    async fn purge_unit(&self) {
        let path = self.unit_path();
        let _ = tokio::fs::remove_file(&path).await;
        if let Some(tok) = &self.cfg.join_token_file {
            let _ = tokio::fs::remove_file(tok).await;
        }
        let _ = self.systemctl(&["daemon-reload"]).await;
    }

    /// `k0s reset` (destructive host-wide cleanup). Returns an error if the reset fails so the
    /// caller can report a partial teardown instead of a false success.
    async fn k0s_reset(&self) -> anyhow::Result<()> {
        let out = Command::new(&self.cfg.k0s_binary)
            .arg("reset")
            .output()
            .await
            .context("failed to run `k0s reset`")?;
        if !out.status.success() {
            anyhow::bail!(
                "`k0s reset` failed ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(())
    }
}

/// Parse a spurd-rendered k0s `ExecStart` value — `<bin> (controller|worker) [--single=true]
/// [--config=P] [--token-file=P]` — back into a [`ClusterConfig`]. Used to re-adopt a running unit
/// after a spurd restart (the unit file is the authoritative record of what spurd started).
fn parse_execstart(exec: &str) -> anyhow::Result<ClusterConfig> {
    let mut parts = exec.split_whitespace();
    let bin = parts.next().context("empty ExecStart")?;
    let subcmd = parts.next().context("ExecStart missing k0s subcommand")?;
    let rest: Vec<&str> = parts.collect();
    let role = match subcmd {
        "worker" => ClusterRole::Worker,
        "controller" if rest.iter().any(|a| a.starts_with("--single")) => ClusterRole::Single,
        "controller" => ClusterRole::Controller,
        other => anyhow::bail!("unexpected k0s subcommand in ExecStart: {other}"),
    };
    let flag_value = |name: &str| {
        rest.iter()
            .find_map(|a| a.strip_prefix(name).map(PathBuf::from))
    };
    let node_ip = rest.iter().find_map(|a| {
        a.strip_prefix("--kubelet-extra-args=--node-ip=")
            .map(String::from)
    });
    Ok(ClusterConfig {
        role,
        k0s_binary: PathBuf::from(bin),
        k0s_config: flag_value("--config="),
        join_token_file: flag_value("--token-file="),
        node_ip,
    })
}

/// Extract the cluster CA (base64) + server URL from an admin kubeconfig YAML. k0s's admin
/// kubeconfig has exactly one cluster, so a dependency-free line scan is sufficient.
fn parse_cluster_ca_server(admin: &str) -> anyhow::Result<(String, String)> {
    let mut ca = None;
    let mut server = None;
    for line in admin.lines() {
        let t = line.trim();
        if let Some(v) = t.strip_prefix("certificate-authority-data:") {
            ca = Some(v.trim().to_string());
        } else if let Some(v) = t.strip_prefix("server:") {
            server = Some(v.trim().to_string());
        }
    }
    match (ca, server) {
        (Some(ca), Some(server)) => Ok((ca, server)),
        _ => anyhow::bail!("admin kubeconfig missing certificate-authority-data or server"),
    }
}

/// Template a namespace-scoped kubeconfig. `name` (the ServiceAccount name) is DNS-safe and the
/// ca/server/token carry no YAML-breaking characters, so plain interpolation is safe.
fn build_scoped_kubeconfig(
    ca: &str,
    server: &str,
    token: &str,
    name: &str,
    namespace: &str,
) -> String {
    format!(
        r#"apiVersion: v1
kind: Config
clusters:
- name: spur
  cluster:
    certificate-authority-data: {ca}
    server: {server}
contexts:
- name: {name}
  context:
    cluster: spur
    namespace: {namespace}
    user: {name}
current-context: {name}
users:
- name: {name}
  user:
    token: {token}
"#
    )
}

#[cfg(test)]
mod scoped_kubeconfig_tests {
    use super::{build_scoped_kubeconfig, parse_cluster_ca_server};

    #[test]
    fn parses_ca_and_server() {
        let admin = "clusters:\n- cluster:\n    certificate-authority-data: QUJD\n    server: https://10.0.0.1:6443\n  name: k0s\n";
        let (ca, server) = parse_cluster_ca_server(admin).unwrap();
        assert_eq!(ca, "QUJD");
        assert_eq!(server, "https://10.0.0.1:6443");
        assert!(parse_cluster_ca_server("apiVersion: v1\n").is_err());
    }

    #[test]
    fn builds_scoped_kubeconfig_yaml() {
        let kc = build_scoped_kubeconfig(
            "QUJD",
            "https://x:6443",
            "tok123",
            "spur-user-alice",
            "spur-acct-physics",
        );
        assert!(kc.starts_with("apiVersion: v1\nkind: Config\n"));
        assert!(kc.contains("namespace: spur-acct-physics"));
        assert!(kc.contains("token: tok123"));
        assert!(kc.contains("certificate-authority-data: QUJD"));
        assert!(kc.contains("current-context: spur-user-alice"));
    }
}

/// Write a bearer-secret file (0600). Content is never logged.
async fn write_secret_file(path: &Path, content: &[u8]) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut opts = tokio::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    // Create with 0600 from the start so a bearer token is never briefly world-readable through
    // the umask-dependent default (the previous write-then-chmod left such a window).
    #[cfg(unix)]
    opts.mode(0o600);
    let mut f = opts.open(path).await?;
    #[cfg(unix)]
    {
        // Also tighten an already-existing file (mode() only applies on create).
        use std::os::unix::fs::PermissionsExt;
        f.set_permissions(std::fs::Permissions::from_mode(0o600))
            .await?;
    }
    f.write_all(content).await?;
    f.flush().await?;
    Ok(())
}

/// Path of the AMD GPU CDI spec written for k0s's containerd.
const CDI_SPEC_PATH: &str = "/etc/cdi/amd.json";

/// Discover local AMD GPUs and write a CDI spec to `/etc/cdi` so k0s's containerd can inject
/// `amd.com/gpu` into pods. No-op when the node has no GPUs — containerd rejects
/// a device-empty spec. Atomic (temp + rename). NOTE: `amd.com/gpu` is also claimed by the ROCm
/// k8s-device-plugin; acceptable for Phase-2 containerd injection — a native spur-device-plugin is
/// a later milestone.
async fn write_cdi_spec() -> anyhow::Result<()> {
    let specs = spur_devices::cdi::discovery::discover_to_cdi();
    if specs.is_empty() {
        info!("no AMD GPUs discovered; not writing a CDI spec");
        return Ok(());
    }
    tokio::fs::create_dir_all("/etc/cdi").await?;
    for (i, spec) in specs.iter().enumerate() {
        spec.validate()
            .map_err(|e| anyhow::anyhow!("invalid CDI spec: {e}"))?;
        let final_path = if i == 0 {
            CDI_SPEC_PATH.to_string()
        } else {
            format!("/etc/cdi/amd-{i}.json")
        };
        let tmp = format!("{final_path}.tmp");
        spec.write_json(Path::new(&tmp))
            .map_err(|e| anyhow::anyhow!("write CDI spec: {e}"))?;
        tokio::fs::rename(&tmp, &final_path).await?;
    }
    info!(count = specs.len(), "wrote AMD GPU CDI spec(s) to /etc/cdi");
    Ok(())
}

/// Remove the CDI spec on teardown (best-effort).
async fn remove_cdi_spec() {
    let _ = tokio::fs::remove_file(CDI_SPEC_PATH).await;
}

/// k0s node status this agent surfaces to the controller on each heartbeat, for `spur_k8s_node_*`
/// metrics. `install_secs` is one-shot: `take_install_secs` returns it once after an install, then
/// clears it, so the controller observes each install duration exactly once.
#[derive(Debug, Default)]
pub struct K0sNodeState {
    unit_active: AtomicBool,
    restart_count: AtomicU64,
    /// Milliseconds; `u64::MAX` sentinel means "nothing to report".
    install_ms: AtomicU64,
}

impl K0sNodeState {
    const NO_INSTALL: u64 = u64::MAX;

    pub fn new() -> Self {
        Self {
            unit_active: AtomicBool::new(false),
            restart_count: AtomicU64::new(0),
            install_ms: AtomicU64::new(Self::NO_INSTALL),
        }
    }

    fn set_unit_active(&self, active: bool) {
        self.unit_active.store(active, Ordering::Relaxed);
    }

    fn set_restart_count(&self, count: u64) {
        self.restart_count.store(count, Ordering::Relaxed);
    }

    fn record_install(&self, dur: Duration) {
        self.install_ms
            .store(dur.as_millis() as u64, Ordering::Relaxed);
    }

    /// Current unit-active + cumulative restarts, plus any pending one-shot install duration in
    /// seconds (0.0 once already drained). Called by the reporter per heartbeat.
    pub fn take_for_heartbeat(&self) -> (bool, u64, f64) {
        let install_ms = self.install_ms.swap(Self::NO_INSTALL, Ordering::Relaxed);
        let install_secs = if install_ms == Self::NO_INSTALL {
            0.0
        } else {
            install_ms as f64 / 1000.0
        };
        (
            self.unit_active.load(Ordering::Relaxed),
            self.restart_count.load(Ordering::Relaxed),
            install_secs,
        )
    }
}

/// RPC-driven owner of this node's k0s component. The spurctld controller
/// (`cluster_k8s`) drives it via the SlurmAgent Start/Stop/GetClusterComponentStatus RPCs. It
/// holds the currently-desired `ClusterSupervisor` and heals its unit from a background loop.
pub struct K0sAgent {
    k0s_binary: PathBuf,
    /// k0s release to install if the binary is missing (a tag or "latest").
    k0s_version: String,
    /// Directory k0s config + the join-token file are written to (`/etc/k0s`).
    config_dir: PathBuf,
    /// Storage provisioner SPUR ships ("local-path" or "none"); [`ClusterConfig::storage_provisioner`].
    storage_provisioner: String,
    /// On-node PV directory for local-path; [`ClusterConfig::local_path_dir`].
    local_path_dir: String,
    /// `[cluster].enabled` from this node's config. When false, `start()` refuses to touch
    /// systemd/k0s — the documented contract for the flag ([`ClusterConfig::enabled`]).
    enabled: bool,
    active: Mutex<Option<ClusterSupervisor>>,
    status: Arc<K0sNodeState>,
}

impl K0sAgent {
    /// Build from the `[cluster]` config so operators can override the k0s version + install path.
    /// (`ClusterConfig::default()` yields the pinned version + `/usr/local/bin/k0s`.)
    pub fn from_config(cfg: &spur_core::config::ClusterConfig) -> Self {
        Self {
            k0s_binary: PathBuf::from(&cfg.k0s_binary),
            k0s_version: cfg.k0s_version.clone(),
            config_dir: PathBuf::from("/etc/k0s"),
            storage_provisioner: cfg.storage_provisioner.clone(),
            local_path_dir: cfg.local_path_dir.clone(),
            enabled: cfg.enabled,
            active: Mutex::new(None),
            status: Arc::new(K0sNodeState::new()),
        }
    }

    /// Shared node-status handle the reporter reads for `spur_k8s_node_*` heartbeat fields.
    pub fn node_state(&self) -> Arc<K0sNodeState> {
        self.status.clone()
    }

    /// Re-adopt an already-running spurd-owned k0s unit at startup. A spurd restart leaves the k0s
    /// systemd unit running (by design) but `active` empty, so `status()` would report `inactive`
    /// and `supervise()` would idle for a unit it owns. This probes the two unit names, and on the
    /// active one rebuilds the `ClusterSupervisor` from the unit's `ExecStart`. No-op (best-effort)
    /// when neither unit is active; call before spawning `supervise()`.
    /// A bare `ClusterSupervisor` for `role` (no config/token) — used to probe/stop the actual
    /// spurd-owned units by name when there is no tracked component.
    fn probe_supervisor(&self, role: ClusterRole) -> ClusterSupervisor {
        ClusterSupervisor::new(ClusterConfig {
            role,
            k0s_binary: self.k0s_binary.clone(),
            k0s_config: None,
            join_token_file: None,
            node_ip: None,
        })
    }

    pub async fn adopt_running_unit(&self) {
        // Controller & Single share k0scontroller.service; the ExecStart disambiguates them.
        for probe_role in [ClusterRole::Controller, ClusterRole::Worker] {
            let probe = self.probe_supervisor(probe_role);
            if !probe.unit_status_is(&["is-active"], "active").await {
                continue;
            }
            match probe.read_cfg_from_unit().await {
                Ok(cfg) => {
                    let role = cfg.role;
                    *self.active.lock().await = Some(ClusterSupervisor::new(cfg));
                    info!(
                        role = role.as_str(),
                        "adopted already-running k0s unit on startup"
                    );
                }
                Err(e) => warn!(
                    unit = probe_role.unit_name(),
                    error = %e,
                    "found an active k0s unit but could not parse its ExecStart; not adopting"
                ),
            }
            return; // at most one k0s unit runs per node
        }
    }

    /// Start (or update) this node's k0s component. Writes the join token (0600) + k0s config to
    /// files, then reconciles the spurd-owned unit. Returns the resulting systemd active-state.
    /// The plaintext `join_token` is never logged.
    pub async fn start(
        &self,
        role: ClusterRole,
        join_token: Option<String>,
        k0s_config: Option<String>,
        node_ip: Option<String>,
    ) -> anyhow::Result<String> {
        // Honor the documented `[cluster].enabled` contract ("spurd never touches systemd/k0s" when
        // false): refuse to install/write/enable/start k0s on a node whose operator disabled it, even
        // if the controller sends a StartClusterComponent RPC. (The handler surfaces this as
        // started=false with this message.)
        if !self.enabled {
            anyhow::bail!(
                "[cluster].enabled is false on this node; refusing to install or start k0s"
            );
        }
        // Make a fresh bare-metal node self-contained: install the pinned/configured k0s if the
        // binary is missing. No-op (no network) when it is already present. A failure here is fatal
        // to start — the systemd unit's ConditionFileIsExecutable would otherwise silently skip.
        let start_began = Instant::now();
        let did_install =
            match spur_update::k0s::ensure_k0s(&self.k0s_version, &self.k0s_binary).await {
                Ok(Some(info)) => {
                    info!(
                        version = %info.version,
                        path = %info.path.display(),
                        "installed k0s"
                    );
                    true
                }
                Ok(None) => false,
                Err(e) => anyhow::bail!(
                    "k0s not present at {} and auto-install (version {}) failed: {e}",
                    self.k0s_binary.display(),
                    self.k0s_version
                ),
            };

        tokio::fs::create_dir_all(&self.config_dir)
            .await
            .with_context(|| format!("create k0s config dir {}", self.config_dir.display()))?;
        let join_token_file = match join_token {
            Some(tok) => {
                let p = self.config_dir.join("token");
                write_secret_file(&p, tok.as_bytes()).await?;
                Some(p)
            }
            None => None,
        };
        let k0s_config = match k0s_config {
            Some(yaml) => {
                let p = self.config_dir.join("k0s.yaml");
                tokio::fs::write(&p, yaml).await?;
                Some(p)
            }
            None => None,
        };
        let cfg = ClusterConfig {
            role,
            k0s_binary: self.k0s_binary.clone(),
            k0s_config,
            join_token_file,
            node_ip,
        };
        // Worker/single nodes run pods, so write the GPU CDI spec before k0s starts (best-effort).
        if role != ClusterRole::Controller {
            if let Err(e) = write_cdi_spec().await {
                warn!(error = %e, "failed to write GPU CDI spec (continuing)");
            }
        }
        // Control-plane nodes (controller/single) run k0s's manifest deployer, so ship the storage
        // addon by writing its manifest there for k0s to apply (best-effort — a bad write must not
        // block the control plane). k0s bundles no storage, so without this a PVC workload hangs.
        if matches!(role, ClusterRole::Controller | ClusterRole::Single)
            && self.storage_provisioner == "local-path"
        {
            if let Err(e) = self.write_local_path_manifest().await {
                warn!(error = %e, "failed to write local-path storage manifest (continuing)");
            }
        }
        let sup = ClusterSupervisor::new(cfg);
        sup.reconcile_once().await?;
        let state = sup.component_state().await;
        self.status.set_unit_active(state == "active");
        self.status
            .set_restart_count(sup.unit_restart_count().await);
        // Record the install→active span only when an install actually happened and the unit came
        // up, so a later start failure never emits a spurious sample.
        if did_install && state == "active" {
            self.status.record_install(start_began.elapsed());
        }
        *self.active.lock().await = Some(sup);
        info!(role = role.as_str(), state = %state, "k0s component started");
        Ok(state)
    }

    /// Ship the local-path storage provisioner: render the manifest with the configured PV directory
    /// and write it into k0s's manifest-deployer dir, which the k0s controller applies + reconciles.
    /// Control-plane only (that is where the manifest deployer runs). Idempotent — overwrites.
    async fn write_local_path_manifest(&self) -> anyhow::Result<()> {
        let dir = Path::new(spur_core::k0s::K0S_MANIFESTS_DIR).join("local-path");
        tokio::fs::create_dir_all(&dir)
            .await
            .with_context(|| format!("create k0s manifests dir {}", dir.display()))?;
        let manifest = spur_core::k0s::k0s_local_path_manifest(&self.local_path_dir);
        let path = dir.join("local-path.yaml");
        tokio::fs::write(&path, manifest)
            .await
            .with_context(|| format!("write {}", path.display()))?;
        info!(
            path = %path.display(),
            data_dir = %self.local_path_dir,
            "shipped local-path storage manifest to k0s manifest deployer"
        );
        Ok(())
    }

    /// Stop + disable this node's component (optionally `k0s reset`). If there is no tracked
    /// component (e.g. spurd restarted and adoption found/parsed nothing), it still stops/resets any
    /// spurd-owned k0s unit present on the host — so `down --reset` is never silently a no-op.
    pub async fn stop(&self, reset: bool) -> anyhow::Result<()> {
        // Take the supervisor out under the lock, then do the (blocking) stop OUTSIDE it so
        // concurrent status/start RPCs aren't serialized behind the seconds-long systemctl/k0s IO.
        let taken = self.active.lock().await.take();
        match taken {
            Some(sup) => sup.stop(reset).await?,
            None => self.stop_untracked(reset).await?,
        }
        // Only after a successful stop — supervise() won't refresh this once `active` is empty.
        self.status.set_unit_active(false);
        remove_cdi_spec().await;
        info!(reset, "k0s component stopped");
        Ok(())
    }

    /// Stop/disable any spurd-owned k0s unit present on the host (a node runs at most one, but both
    /// names are disabled idempotently), then `k0s reset` once if requested. Used when `active` is
    /// empty so a lost in-memory state can't leave a running k0s unit behind. A failed reset is
    /// surfaced so `down --reset` reports the partial teardown rather than a false success.
    async fn stop_untracked(&self, reset: bool) -> anyhow::Result<()> {
        for role in [ClusterRole::Controller, ClusterRole::Worker] {
            self.probe_supervisor(role).disable_unit().await;
        }
        if reset {
            // Purge UNCONDITIONALLY, even if `k0s reset` fails (broken/half-installed k0s binary):
            // a surviving token was minted against the torn-down CA and makes the node fail its next
            // join with a kubernetes-ca verification error. Surface the reset failure only after
            // purging. The probe supervisor carries no token path, so also remove the token directly.
            let reset_result = self
                .probe_supervisor(ClusterRole::Controller)
                .k0s_reset()
                .await;
            for role in [ClusterRole::Controller, ClusterRole::Worker] {
                self.probe_supervisor(role).purge_unit().await;
            }
            let _ = tokio::fs::remove_file(self.config_dir.join("token")).await;
            reset_result?;
        }
        Ok(())
    }

    /// Mint a k0s join token on this (control-plane) node via `k0s token create`. Returns the
    /// plaintext token (bearer secret — never logged). Errors on a non-control-plane node or before
    /// the k0s controller API is reachable; the controller retries on the next reconcile tick.
    pub async fn create_join_token(
        &self,
        role: &str,
        expiry_seconds: u64,
    ) -> anyhow::Result<String> {
        let mut cmd = Command::new(&self.k0s_binary);
        cmd.args(["token", "create", "--role", role]);
        if expiry_seconds > 0 {
            cmd.args(["--expiry", &format!("{expiry_seconds}s")]);
        }
        let out = cmd.output().await?;
        if !out.status.success() {
            anyhow::bail!(
                "k0s token create --role {role} failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        let token = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if token.is_empty() {
            anyhow::bail!("k0s token create returned an empty token");
        }
        info!(role, "minted k0s join token"); // token value never logged
        Ok(token)
    }

    /// Read the admin kubeconfig on this (control-plane) node via `k0s kubeconfig admin`. Returns
    /// the YAML. Errors on a non-control-plane node (no local admin credentials).
    pub async fn admin_kubeconfig(&self) -> anyhow::Result<String> {
        let out = Command::new(&self.k0s_binary)
            .args(["kubeconfig", "admin"])
            .output()
            .await?;
        if !out.status.success() {
            anyhow::bail!(
                "k0s kubeconfig admin failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        let kubeconfig = String::from_utf8_lossy(&out.stdout).to_string();
        if kubeconfig.trim().is_empty() {
            anyhow::bail!("k0s kubeconfig admin returned empty output");
        }
        Ok(kubeconfig)
    }

    /// Mint a namespace-scoped kubeconfig on this (control-plane) node: ensure `service_account`
    /// exists in `namespace`, mint a bound token for it (`k0s kubectl create token`), and template a
    /// kubeconfig with the admin cluster CA + server scoped to that namespace. The namespace + its
    /// RBAC are expected to exist (created by the operator's quota reconciler). Token never logged.
    pub async fn user_kubeconfig(
        &self,
        user: &str,
        namespace: &str,
        service_account: &str,
    ) -> anyhow::Result<String> {
        // Fail closed: this is the token-minting boundary. Empty inputs must never reach kubectl,
        // where they could scope a token to an unintended namespace/SA.
        if user.is_empty() || namespace.is_empty() || service_account.is_empty() {
            anyhow::bail!(
                "user_kubeconfig requires non-empty user, namespace, and service_account"
            );
        }
        // Ensure the ServiceAccount exists — idempotent (an already-existing SA is fine).
        let create = Command::new(&self.k0s_binary)
            .args([
                "kubectl",
                "create",
                "serviceaccount",
                service_account,
                "-n",
                namespace,
            ])
            .output()
            .await?;
        if !create.status.success() {
            let err = String::from_utf8_lossy(&create.stderr);
            if !err.contains("AlreadyExists") && !err.contains("already exists") {
                anyhow::bail!(
                    "create serviceaccount {service_account} in {namespace} failed: {}",
                    err.trim()
                );
            }
        }
        // Mint a bound (rotatable) token for the ServiceAccount.
        let tok = Command::new(&self.k0s_binary)
            .args([
                "kubectl",
                "create",
                "token",
                service_account,
                "-n",
                namespace,
                "--duration=8760h",
            ])
            .output()
            .await?;
        if !tok.status.success() {
            anyhow::bail!(
                "create token for {service_account} in {namespace} failed: {}",
                String::from_utf8_lossy(&tok.stderr).trim()
            );
        }
        let token = String::from_utf8_lossy(&tok.stdout).trim().to_string();
        if token.is_empty() {
            anyhow::bail!("k0s kubectl create token returned empty output");
        }
        // Reuse the admin kubeconfig for the cluster CA + server URL.
        let (ca, server) = parse_cluster_ca_server(&self.admin_kubeconfig().await?)?;
        info!(user, namespace, service_account, "minted scoped kubeconfig");
        Ok(build_scoped_kubeconfig(
            &ca,
            &server,
            &token,
            service_account,
            namespace,
        ))
    }

    /// Cordon then drain a k8s node via `k0s kubectl` (run on a control-plane node). Cordon marks it
    /// unschedulable; drain evicts pods PDB-aware, skipping DaemonSets and deleting emptyDir data.
    /// `force` adds `--force --disable-eviction` (bypass PDBs — data-loss risk) for a stuck drain.
    /// Returns Ok(()) only when cordon + drain succeed. Deleting the Node object is a separate step
    /// (`delete_node`) the controller runs AFTER stopping the component, so the kubelet can't
    /// re-register an uncordoned node in the gap.
    pub async fn drain_node(
        &self,
        node: &str,
        timeout_secs: u32,
        force: bool,
    ) -> anyhow::Result<()> {
        if node.is_empty() {
            anyhow::bail!("drain_node requires a non-empty node name");
        }
        let cordon = Command::new(&self.k0s_binary)
            .args(["kubectl", "cordon", node])
            .output()
            .await?;
        if !cordon.status.success() {
            anyhow::bail!(
                "k0s kubectl cordon {node} failed: {}",
                String::from_utf8_lossy(&cordon.stderr).trim()
            );
        }
        let timeout = if timeout_secs == 0 { 120 } else { timeout_secs };
        let mut args: Vec<String> = vec![
            "kubectl".into(),
            "drain".into(),
            node.to_string(),
            "--ignore-daemonsets".into(),
            "--delete-emptydir-data".into(),
            format!("--timeout={timeout}s"),
        ];
        if force {
            args.push("--force".into());
            args.push("--disable-eviction".into());
        }
        let drain = Command::new(&self.k0s_binary).args(&args).output().await?;
        if !drain.status.success() {
            anyhow::bail!(
                "k0s kubectl drain {node} failed: {}",
                String::from_utf8_lossy(&drain.stderr).trim()
            );
        }
        info!(node, force, "k8s node cordoned + drained");
        Ok(())
    }

    /// Delete a k8s Node object via `k0s kubectl delete node --ignore-not-found` (control-plane node).
    /// The controller calls this only AFTER the node's component is stopped, so the kubelet is gone
    /// and cannot re-register a fresh (uncordoned) Node. Tolerant of an absent node so a re-run is
    /// idempotent.
    pub async fn delete_node(&self, node: &str) -> anyhow::Result<()> {
        if node.is_empty() {
            anyhow::bail!("delete_node requires a non-empty node name");
        }
        let del = Command::new(&self.k0s_binary)
            .args(["kubectl", "delete", "node", node, "--ignore-not-found"])
            .output()
            .await?;
        if !del.status.success() {
            anyhow::bail!(
                "k0s kubectl delete node {node} failed: {}",
                String::from_utf8_lossy(&del.stderr).trim()
            );
        }
        info!(node, "k8s node object deleted");
        Ok(())
    }

    /// (role, active_state, enabled) for the node's component. When there is no tracked component,
    /// it probes the actual spurd-owned units so a restarted spurd (before/without adoption) still
    /// reports the truth instead of a false `inactive`.
    pub async fn status(&self) -> (String, String, bool) {
        // Snapshot out of the lock, then query systemctl OUTSIDE it (avoids serializing RPCs).
        let tracked = self.active.lock().await.clone();
        if let Some(sup) = tracked {
            let enabled = sup.unit_status_is(&["is-enabled"], "enabled").await;
            return (
                sup.role().as_str().to_string(),
                sup.component_state().await,
                enabled,
            );
        }
        // Untracked: report a running spurd-owned unit if there is one (Controller/Single share a
        // unit name — reported as "controller").
        for role in [ClusterRole::Worker, ClusterRole::Controller] {
            let probe = self.probe_supervisor(role);
            let state = probe.component_state().await;
            if state != "inactive" && state != "unknown" {
                let enabled = probe.unit_status_is(&["is-enabled"], "enabled").await;
                return (role.as_str().to_string(), state, enabled);
            }
        }
        (String::new(), "inactive".to_string(), false)
    }

    /// Background heal loop: keep the active component's unit reconciled. Spawned from spurd
    /// `main()`; idle (no-op) until a `start` sets an active component.
    pub async fn supervise(self: Arc<Self>) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            // Snapshot the supervisor, then reconcile OUTSIDE the lock (reconcile shells out + does
            // filesystem IO; holding the lock would serialize concurrent start/stop/status RPCs).
            let sup = self.active.lock().await.clone();
            if let Some(sup) = sup {
                if let Err(e) = sup.reconcile_once().await {
                    warn!(error = %e, "k0s component reconcile failed");
                }
                self.status
                    .set_unit_active(sup.unit_status_is(&["is-active"], "active").await);
                self.status
                    .set_restart_count(sup.unit_restart_count().await);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_state_install_duration_is_one_shot() {
        let state = K0sNodeState::new();
        state.set_unit_active(true);
        state.set_restart_count(3);
        state.record_install(Duration::from_millis(2500));

        let (active, restarts, install) = state.take_for_heartbeat();
        assert!(active);
        assert_eq!(restarts, 3);
        assert_eq!(install, 2.5);

        // Drained: a second read reports 0 install seconds but retains active/restarts.
        let (active2, restarts2, install2) = state.take_for_heartbeat();
        assert!(active2);
        assert_eq!(restarts2, 3);
        assert_eq!(install2, 0.0);
    }

    #[test]
    fn node_state_defaults_report_nothing_pending() {
        let (active, restarts, install) = K0sNodeState::new().take_for_heartbeat();
        assert!(!active);
        assert_eq!(restarts, 0);
        assert_eq!(install, 0.0);
    }

    #[tokio::test]
    async fn purge_unit_removes_unit_file_and_token() {
        let dir = tempfile::TempDir::new().unwrap();
        let unit_dir = dir.path().join("systemd");
        tokio::fs::create_dir_all(&unit_dir).await.unwrap();
        let token = dir.path().join("token");
        tokio::fs::write(&token, b"secret").await.unwrap();

        let sup = ClusterSupervisor {
            cfg: ClusterConfig {
                role: ClusterRole::Worker,
                k0s_binary: PathBuf::from("/usr/local/bin/k0s"),
                k0s_config: None,
                join_token_file: Some(token.clone()),
                node_ip: None,
            },
            unit_dir: unit_dir.clone(),
        };
        // Simulate the installed unit file.
        let unit_path = unit_dir.join(sup.unit_file_name());
        tokio::fs::write(&unit_path, b"[Unit]\n").await.unwrap();

        sup.purge_unit().await;

        assert!(
            !unit_path.exists(),
            "purge_unit must remove the spurd-owned unit file"
        );
        assert!(
            !token.exists(),
            "purge_unit must remove the join token so the node can't rejoin"
        );
    }

    #[test]
    fn renders_controller_unit_with_validated_directives() {
        let sup = ClusterSupervisor::new(ClusterConfig {
            role: ClusterRole::Single,
            k0s_binary: PathBuf::from("/usr/local/bin/k0s"),
            k0s_config: None,
            join_token_file: None,
            node_ip: None,
        });
        let unit = sup.render_unit();
        assert!(unit.contains("ExecStart=/usr/local/bin/k0s controller --single=true"));
        // The directives that make a spurd-owned k0s unit behave correctly.
        assert!(unit.contains("KillMode=process"));
        assert!(unit.contains("Delegate=yes"));
        assert!(unit.contains("Restart=always"));
        // k0s does not sd_notify: the unit must NOT be Type=notify (default simple is correct).
        assert!(!unit.contains("Type=notify"));
        assert_eq!(sup.unit_file_name(), "k0scontroller.service");
    }

    #[test]
    fn worker_unit_carries_config_and_token() {
        let sup = ClusterSupervisor::new(ClusterConfig {
            role: ClusterRole::Worker,
            k0s_binary: PathBuf::from("/usr/local/bin/k0s"),
            k0s_config: Some(PathBuf::from("/etc/k0s/k0s.yaml")),
            join_token_file: Some(PathBuf::from("/etc/k0s/token")),
            node_ip: Some("10.44.0.2".to_string()),
        });
        let unit = sup.render_unit();
        assert!(unit.contains("k0s worker"));
        assert!(unit.contains("--config=/etc/k0s/k0s.yaml"));
        assert!(unit.contains("--token-file=/etc/k0s/token"));
        assert!(unit.contains("--kubelet-extra-args=--node-ip=10.44.0.2"));
        assert_eq!(sup.unit_file_name(), "k0sworker.service");
    }

    #[test]
    fn parse_execstart_recovers_role_and_paths() {
        let single = parse_execstart("/usr/local/bin/k0s controller --single=true").unwrap();
        assert_eq!(single.role, ClusterRole::Single);
        assert_eq!(single.k0s_binary, PathBuf::from("/usr/local/bin/k0s"));
        assert!(single.k0s_config.is_none() && single.join_token_file.is_none());

        let ctrl = parse_execstart("/opt/k0s controller --config=/etc/k0s/k0s.yaml").unwrap();
        assert_eq!(ctrl.role, ClusterRole::Controller);
        assert_eq!(ctrl.k0s_binary, PathBuf::from("/opt/k0s"));
        assert_eq!(ctrl.k0s_config, Some(PathBuf::from("/etc/k0s/k0s.yaml")));

        let worker = parse_execstart(
            "/usr/local/bin/k0s worker --token-file=/etc/k0s/token --kubelet-extra-args=--node-ip=10.44.0.2",
        )
        .unwrap();
        assert_eq!(worker.role, ClusterRole::Worker);
        assert_eq!(
            worker.join_token_file,
            Some(PathBuf::from("/etc/k0s/token"))
        );
        assert_eq!(worker.node_ip.as_deref(), Some("10.44.0.2"));

        assert!(parse_execstart("").is_err());
        assert!(parse_execstart("/usr/local/bin/k0s bogus").is_err());
    }

    /// The adopt path must re-render an *identical* unit, so a rendered ExecStart must parse back
    /// into a cfg whose re-render matches (otherwise reconcile_once would rewrite the file).
    #[test]
    fn execstart_render_parse_roundtrip() {
        for cfg in [
            ClusterConfig {
                role: ClusterRole::Single,
                k0s_binary: PathBuf::from("/usr/local/bin/k0s"),
                k0s_config: None,
                join_token_file: None,
                node_ip: None,
            },
            ClusterConfig {
                role: ClusterRole::Worker,
                k0s_binary: PathBuf::from("/usr/local/bin/k0s"),
                k0s_config: Some(PathBuf::from("/etc/k0s/k0s.yaml")),
                join_token_file: Some(PathBuf::from("/etc/k0s/token")),
                node_ip: Some("10.44.0.2".to_string()),
            },
        ] {
            let unit = ClusterSupervisor::new(cfg.clone()).render_unit();
            let exec = unit
                .lines()
                .find_map(|l| l.trim().strip_prefix("ExecStart="))
                .unwrap();
            let parsed = parse_execstart(exec).unwrap();
            assert_eq!(parsed.role, cfg.role);
            assert_eq!(parsed.k0s_binary, cfg.k0s_binary);
            assert_eq!(parsed.k0s_config, cfg.k0s_config);
            assert_eq!(parsed.join_token_file, cfg.join_token_file);
            assert_eq!(parsed.node_ip, cfg.node_ip);
            // Re-rendered unit is byte-identical -> reconcile stays a no-op.
            assert_eq!(ClusterSupervisor::new(parsed).render_unit(), unit);
        }
    }

    /// Live de-risk: exercises the real reconcile against systemd + k0s. Gated behind
    /// `SPUR_CLUSTER_DERISK=1` and requires root on a Linux host with systemd + k0s installed
    /// (single-node). Run on shark-a:
    ///   sudo -E ~/.cargo/bin/cargo test -p spurd cluster::tests::derisk -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "requires root + systemd + k0s; run explicitly on a Linux host"]
    async fn derisk_supervise_recovers_unit() {
        if std::env::var_os("SPUR_CLUSTER_DERISK").is_none() {
            eprintln!("skip: set SPUR_CLUSTER_DERISK=1 and run as root to exercise this");
            return;
        }
        let sup = ClusterSupervisor::new(ClusterConfig {
            role: ClusterRole::Single,
            k0s_binary: PathBuf::from("/usr/local/bin/k0s"),
            k0s_config: None,
            join_token_file: None,
            node_ip: None,
        });

        // 1. Fresh apply: unit written, enabled, active.
        sup.reconcile_once().await.expect("initial reconcile");
        assert!(
            sup.unit_status_is(&["is-active"], "active").await,
            "active after reconcile"
        );
        assert!(
            sup.unit_status_is(&["is-enabled"], "enabled").await,
            "enabled after reconcile"
        );

        // 2. Out-of-band disable+stop -> reconcile must restore (Restart= does NOT cover this).
        sup.systemctl(&["disable", "--now", &sup.unit_file_name()])
            .await
            .expect("disable");
        assert!(
            !sup.unit_status_is(&["is-active"], "active").await,
            "stopped after disable --now"
        );
        sup.reconcile_once().await.expect("reconcile after disable");
        assert!(
            sup.unit_status_is(&["is-active"], "active").await,
            "reconcile restarts a stopped unit"
        );
        assert!(
            sup.unit_status_is(&["is-enabled"], "enabled").await,
            "reconcile re-enables"
        );

        // 3. Delete the unit file -> reconcile must re-write it.
        tokio::fs::remove_file(sup.unit_path()).await.ok();
        sup.systemctl(&["daemon-reload"]).await.ok();
        sup.reconcile_once().await.expect("reconcile after delete");
        assert!(
            sup.unit_path().exists(),
            "reconcile re-writes a deleted unit file"
        );
        assert!(
            sup.unit_status_is(&["is-active"], "active").await,
            "active again after re-write"
        );
    }

    #[tokio::test]
    async fn reset_and_purge_removes_token_even_when_k0s_reset_fails() {
        // Regression: `down --reset` must purge the spurd-owned unit + join token even if `k0s reset`
        // fails (e.g. a broken/half-installed k0s binary). A surviving token was minted against the
        // torn-down CA, so the node fails its next join with a kubernetes-ca verification error. The
        // reset failure must still be surfaced (Err), but only AFTER the purge. Drives
        // `reset_and_purge` directly so no systemd/k0s host state is touched (`k0s reset` fails via a
        // non-existent binary; `purge_unit`'s trailing daemon-reload is best-effort/ignored).
        let dir = tempfile::TempDir::new().unwrap();
        let unit_dir = dir.path().join("systemd");
        tokio::fs::create_dir_all(&unit_dir).await.unwrap();
        let token = dir.path().join("token");
        tokio::fs::write(&token, b"stale-ca-token").await.unwrap();

        let sup = ClusterSupervisor {
            cfg: ClusterConfig {
                role: ClusterRole::Worker,
                // Non-existent binary: `k0s reset` fails deterministically without touching the host.
                k0s_binary: dir.path().join("no-such-k0s"),
                k0s_config: None,
                join_token_file: Some(token.clone()),
                node_ip: None,
            },
            unit_dir: unit_dir.clone(),
        };
        let unit_path = unit_dir.join(sup.unit_file_name());
        tokio::fs::write(&unit_path, b"[Unit]\n").await.unwrap();

        let result = sup.reset_and_purge().await;

        assert!(
            result.is_err(),
            "a failed `k0s reset` must be surfaced, not swallowed"
        );
        assert!(
            !token.exists(),
            "the join token must be purged even when `k0s reset` fails"
        );
        assert!(
            !unit_path.exists(),
            "the spurd-owned unit must be purged even when `k0s reset` fails"
        );
    }

    #[tokio::test]
    async fn stop_clears_unit_active_even_with_no_tracked_supervisor() {
        let agent = K0sAgent::from_config(&spur_core::config::ClusterConfig::default());
        agent.status.set_unit_active(true);

        agent.stop(false).await.unwrap();

        let (active, _, _) = agent.status.take_for_heartbeat();
        assert!(
            !active,
            "stop() must clear unit_active so heartbeats stop reporting node_up=1"
        );
    }

    #[tokio::test]
    async fn user_kubeconfig_fails_closed_on_empty_inputs() {
        let agent = K0sAgent::from_config(&spur_core::config::ClusterConfig::default());
        for (user, ns, sa) in [
            ("", "spur-acct-x", "spur-user-y"),
            ("u", "", "spur-user-y"),
            ("u", "spur-acct-x", ""),
        ] {
            let err = agent
                .user_kubeconfig(user, ns, sa)
                .await
                .expect_err("empty input must fail closed before minting a token");
            assert!(
                err.to_string().contains("non-empty"),
                "unexpected error: {err}"
            );
        }
    }
}
